import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models.patch_encoder      import PatchEmbedding
from models.mamba_simple       import MambaClassifier
from models.neural_ode_router  import NeuralODERouter, FixedRouterHilbert
from models.continuous_graph_mamba import ContinuousGraphMambaClassifier as CGMamba
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba

class ScatteredCIFAR(Dataset):
    def __init__(self, train=True, canvas_size=128, patch_size=8, noise_std=0.1, seed=42):
        super().__init__()
        self.canvas_size = canvas_size
        self.patch_size  = patch_size
        self.noise_std   = noise_std
        self.grid_size   = canvas_size // patch_size
        self.n_patches   = self.grid_size ** 2       

        self.cifar = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
            ])
        )

        self.image_grid = 32 // patch_size
        self.n_image_patches = self.image_grid ** 2

        self.rng = np.random.RandomState(seed if not train else seed + 1)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx]   
        canvas = torch.randn(3, self.canvas_size, self.canvas_size) * self.noise_std
        p = self.patch_size
        image_patches = []
        for r in range(self.image_grid):
            for c in range(self.image_grid):
                patch = img[:, r*p:(r+1)*p, c*p:(c+1)*p] 
                image_patches.append(patch)

        all_positions = list(range(self.n_patches))
        self.rng.shuffle(all_positions)
        image_positions = sorted(all_positions[:self.n_image_patches])

        for i, pos in enumerate(image_positions):
            row = pos // self.grid_size
            col = pos % self.grid_size
            canvas[:, row*p:(row+1)*p, col*p:(col+1)*p] = image_patches[i]

        return canvas, label

class HilbertMambaScattered(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(img_size=cfg.canvas_size, patch_size=cfg.patch_size, in_channels=3, d_embed=cfg.d_embed)
        grid = cfg.canvas_size // cfg.patch_size
        self.router = FixedRouterHilbert(grid, grid)
        self.mamba = MambaClassifier(d_model=cfg.d_embed, n_classes=10, n_layers=cfg.n_mamba_layers, d_state=cfg.d_state)
    def forward(self, x):
        emb = self.embedder(x)
        emb, _ = self.router(emb)
        return self.mamba(emb)

class ODEMambaScattered(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(img_size=cfg.canvas_size, patch_size=cfg.patch_size, in_channels=3, d_embed=cfg.d_embed)
        self.router = NeuralODERouter(d_embed=cfg.d_embed, d_ff=cfg.ode_d_ff, tau=cfg.tau, solver=cfg.ode_solver, n_steps=cfg.ode_steps)
        self.mamba = MambaClassifier(d_model=cfg.d_embed, n_classes=10, n_layers=cfg.n_mamba_layers, d_state=cfg.d_state)
    def forward(self, x):
        emb = self.embedder(x)
        emb, scores = self.router(emb)
        return self.mamba(emb)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
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
        imgs, labels = imgs.to(device), labels.to(device)
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
    n_patches = (cfg.canvas_size // cfg.patch_size) ** 2
    print(f"  Patches:  {n_patches} per image ({cfg.canvas_size//cfg.patch_size}x{cfg.canvas_size//cfg.patch_size} grid)")
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

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{name}_best.pt"))
        print(f"  Epoch {epoch:03d} | Train {tr_acc:5.1f}% | Val {vl_acc:5.1f}% | Time {elapsed:.1f}s")
    print(f"\n  ✓ Best Val Acc: {best_val_acc:.2f}%")

def main():
    p = argparse.ArgumentParser(description="Scattered CIFAR Hypothesis Test")
    p.add_argument('--mode', choices=['hilbert', 'ode', 'graph_ode', 'spatial', 'all'], default='all')
    p.add_argument('--canvas_size', type=int, default=128, help="Canvas size (128 → 16x16=256 patches)")
    p.add_argument('--patch_size',  type=int, default=8)
    p.add_argument('--noise_std',   type=float, default=0.1)
    p.add_argument('--d_embed',        type=int, default=64)
    p.add_argument('--d_state',        type=int, default=16)
    p.add_argument('--n_mamba_layers', type=int, default=2)
    p.add_argument('--ode_d_ff',   type=int,   default=64)
    p.add_argument('--tau',        type=float, default=0.5)
    p.add_argument('--ode_solver', default='rk4', choices=['euler', 'rk4', 'dopri5'])
    p.add_argument('--ode_steps',  type=int, default=10)
    p.add_argument('--K_steps',    type=int, default=3)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--save_dir',     default='.')
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_patches = (cfg.canvas_size // cfg.patch_size) ** 2
    print(f"\nDevice: {device}")
    print(f"  Canvas: {cfg.canvas_size}x{cfg.canvas_size}")
    print(f"  Patch:  {cfg.patch_size}x{cfg.patch_size}")
    print(f"  Image patches: 16 / {n_patches} ({100*16/n_patches:.1f}% signal, rest noise)\n")

    train_ds = ScatteredCIFAR(train=True, canvas_size=cfg.canvas_size, patch_size=cfg.patch_size, noise_std=cfg.noise_std, seed=cfg.seed)
    val_ds = ScatteredCIFAR(train=False, canvas_size=cfg.canvas_size, patch_size=cfg.patch_size, noise_std=cfg.noise_std, seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    if cfg.mode in ('hilbert', 'all'):
        model_h = HilbertMambaScattered(cfg).to(device)
        train_model('HilbertMamba', model_h, train_loader, val_loader, cfg, device)

    if cfg.mode in ('graph_ode', 'all'):
        model_g = CGMamba(cfg).to(device)
        train_model('CGMamba', model_g, train_loader, val_loader, cfg, device)

    if cfg.mode in ('spatial', 'all'):
        model_s = CSMamba(cfg).to(device)
        train_model('CSMamba', model_s, train_loader, val_loader, cfg, device)

if __name__ == '__main__':
    main()
