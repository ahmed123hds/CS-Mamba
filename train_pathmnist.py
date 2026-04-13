import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import medmnist
from medmnist import INFO

sys.path.insert(0, os.path.dirname(__file__))

from models.patch_encoder      import PatchEmbedding
from models.mamba_simple       import MambaClassifier
from models.continuous_graph_mamba import ContinuousGraphMambaClassifier as CGMamba
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba

class HilbertMambaPath(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(img_size=cfg.img_size, patch_size=cfg.patch_size, in_channels=3, d_embed=cfg.d_embed)
        grid = cfg.img_size // cfg.patch_size
        self.router = FixedRouterHilbert(grid, grid)
        self.mamba = MambaClassifier(d_model=cfg.d_embed, n_classes=cfg.n_classes, n_layers=cfg.n_mamba_layers, d_state=cfg.d_state)
    def forward(self, x):
        emb = self.embedder(x)
        emb, _ = self.router(emb)
        return self.mamba(emb)

def get_dataloaders(cfg):
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    # Upscale 28x28 to 128x128 so it perfectly mimics our 16x16 scattered patch architecture
    train_tf = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    val_tf = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    train_ds = DataClass(split='train', transform=train_tf, download=True)
    val_ds   = DataClass(split='val', transform=val_tf, download=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).squeeze().long()
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).squeeze().long()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total

def train_model(name, model, train_loader, val_loader, cfg, device):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Params:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Classes:  {cfg.n_classes} (PathMNIST)")
    n_patches = (cfg.img_size // cfg.patch_size) ** 2
    print(f"  Patches:  {n_patches} per image ({cfg.img_size//cfg.patch_size}x{cfg.img_size//cfg.patch_size} grid)")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    if hasattr(torch, 'compile'):
        try:
            print("  [Auto-fusing Triton Kernels...]")
            model = torch.compile(model)
        except Exception as e:
            pass

    history = {'train_acc': [], 'val_acc': [], 'epoch_time': []}
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['epoch_time'].append(elapsed)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), f"{name}_best_pathmnist.pt")

        print(f"  Epoch {epoch:03d}/{cfg.epochs} | Train {tr_acc:5.1f}% | Val {vl_acc:5.1f}% | Time {elapsed:.1f}s")
    
    print(f"\n  ✓ Best Val Acc: {best_val_acc:.2f}%")
    return history


def parse_args():
    p = argparse.ArgumentParser(description="PathMNIST Medical WSI-Proxy Training")
    p.add_argument('--mode', choices=['hilbert', 'graph_ode', 'spatial', 'all'], default='all')
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--img_size',     type=int, default=128)
    p.add_argument('--patch_size',   type=int, default=8)
    p.add_argument('--n_classes',    type=int, default=9)
    p.add_argument('--d_embed',        type=int,   default=128)
    p.add_argument('--d_state',        type=int,   default=16)
    p.add_argument('--n_mamba_layers', type=int,   default=2)
    p.add_argument('--K_steps',    type=int,   default=3)
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    train_loader, val_loader = get_dataloaders(cfg)
    results = {}

    if cfg.mode in ('hilbert', 'all'):
        model_h = HilbertMambaPath(cfg).to(device)
        from models.neural_ode_router import FixedRouterHilbert
        results['HilbertMamba'] = train_model('HilbertMamba', model_h, train_loader, val_loader, cfg, device)

    if cfg.mode in ('graph_ode', 'all'):
        setattr(cfg, 'canvas_size', cfg.img_size) 
        model_g = CGMamba(cfg).to(device)
        results['CGMamba'] = train_model('CGMamba', model_g, train_loader, val_loader, cfg, device)

    if cfg.mode in ('spatial', 'all'):
        setattr(cfg, 'canvas_size', cfg.img_size) 
        model_s = CSMamba(cfg).to(device)
        results['CSMamba'] = train_model('CSMamba', model_s, train_loader, val_loader, cfg, device)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  FINAL COMPARISON - PathMNIST WSI-Proxy")
        print(f"{'='*60}")
        for name, hist in results.items():
            avg_time = np.mean(hist['epoch_time'])
            print(f"  {name:20s}  Best Val: {hist['best_val_acc']:5.2f}%  Avg epoch: {avg_time:.1f}s")
    print("\nDone.")

if __name__ == '__main__':
    from models.neural_ode_router import FixedRouterHilbert
    main()
