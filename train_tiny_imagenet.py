"""
Tiny-ImageNet Training Script for CS-Mamba
===========================================
Implements the full SOTA regularization stack used by:
  - DeiT (Touvron et al., 2021)  : Mixup, CutMix, RandAugment, Label Smoothing
  - VMamba (Liu et al., 2024)    : AdamW + Cosine LR + Weight Decay 0.05
  - Swin Transformer (2021)      : Stochastic Depth, Layer Scale
  - Mamba (Gu et al., 2023)      : Gradient Clipping 1.0

Run:
  python train_tiny_imagenet.py --mode spatial     # CS-Mamba only
  python train_tiny_imagenet.py --mode hilbert     # Hilbert Mamba baseline
  python train_tiny_imagenet.py --mode all         # Both head-to-head
"""

import os, time, math, argparse, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms

from datasets import load_dataset

# ─────────────────────────────────────────────────────────
#  Model Imports
# ─────────────────────────────────────────────────────────
from mamba_ssm import Mamba
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba
from models.patch_encoder import PatchEmbedding

# Hilbert Mamba baseline: cfg-compatible (no separate file needed)
class HilbertMamba(nn.Module):
    """Standard 1D Vision Mamba with official CUDA-optimized Mamba kernel."""
    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(
            img_size=getattr(cfg, 'canvas_size', cfg.img_size),
            patch_size=cfg.patch_size, in_channels=3, d_embed=cfg.d_embed)
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=cfg.d_embed, d_state=cfg.d_state, d_conv=4, expand=2)
            for _ in range(cfg.n_mamba_layers)
        ])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_embed) for _ in range(cfg.n_mamba_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_embed)
        self.head = nn.Linear(cfg.d_embed, getattr(cfg, 'n_classes', 200))

    def forward(self, x):
        x = self.embedder(x)
        for norm, mamba in zip(self.norms, self.mamba_blocks):
            x = x + mamba(norm(x))
        x = self.final_norm(x)
        return self.head(x.mean(dim=1))


# ─────────────────────────────────────────────────────────
#  SOTA Regularization: Mixup + CutMix
#  Used by DeiT, VMamba, Swin, ViT-*
# ─────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.8, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """CutMix: crops a random rectangle from another image."""
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(device)

    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = item['image']
        label = item['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(cfg):
    print("  [Loading Tiny-ImageNet via HuggingFace...]")
    ds = load_dataset("Maysee/tiny-imagenet")

    # ImageNet normalisation (identical for Tiny-ImageNet)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # ── SOTA Training Augmentation (DeiT / VMamba stack) ────────────
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.67, 1.0)),  # prevents trivial cropping
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),   # DeiT / ViT standard
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),  # Swin Transformer
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = TinyImageNetDataset(ds['train'], transform_train)
    val_ds   = TinyImageNetDataset(ds['valid'], transform_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────
#  One Epoch: Train
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, cfg):
    model.train()
    correct = total = total_loss = 0
    t0 = time.time()
    use_mixup  = random.random() < 0.5   # 50% epochs use Mixup
    use_cutmix = not use_mixup            # alternating 50% use CutMix

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Apply Mixup or CutMix
        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.8, device=device)
        elif use_cutmix:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=1.0, device=device)

        with autocast('cuda'):
            outs = model(imgs)
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outs, y_a, y_b, lam)
            else:
                loss = criterion(outs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping (Mamba standard: max_norm = 1.0)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, 100.0 * correct / total, time.time() - t0


# ─────────────────────────────────────────────────────────
#  One Epoch: Validate
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast('cuda'):
            outs = model(imgs)
            loss = criterion(outs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────────────────
def train_model(name, model, train_loader, val_loader, cfg, device):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Params:   {n_params:,}")
    print(f"  Dataset:  Tiny-ImageNet (200 classes, 100K images)")
    print(f"  Patches:  {(cfg.img_size//cfg.patch_size)**2} per image ({cfg.img_size//cfg.patch_size}x{cfg.img_size//cfg.patch_size} grid)")
    print(f"{'='*60}")

    # Label Smoothing (DeiT, VMamba, Swin all use 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW with strong weight decay (VMamba: 0.05)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.lr,
                            weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.999),
                            eps=1e-8)

    # Warmup + Cosine Annealing (DeiT standard)
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler('cuda')

    best_acc = 0.0
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, t_ep = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, cfg)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), f"best_{name}.pt")

        print(f"  Epoch {ep:03d}/{cfg.epochs} | "
              f"Train {tr_acc:5.1f}% | Val {vl_acc:5.1f}% | "
              f"Time {t_ep:4.1f}s | LR {scheduler.get_last_lr()[0]:.2e}")

    print(f"\n  ✓ Best Val Acc: {best_acc:.2f}%\n")
    return best_acc


# ─────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Tiny-ImageNet CS-Mamba Benchmarking")
    p.add_argument('--mode',          choices=['hilbert', 'spatial', 'all'], default='all')
    p.add_argument('--img_size',      type=int,   default=64)
    p.add_argument('--patch_size',    type=int,   default=4)     # → 16×16 = 256 patches
    p.add_argument('--n_classes',     type=int,   default=200)
    p.add_argument('--d_embed',       type=int,   default=192)   # ViT-Tiny width
    p.add_argument('--d_state',       type=int,   default=16)
    p.add_argument('--n_mamba_layers',type=int,   default=8)
    p.add_argument('--K_steps',       type=int,   default=3)
    p.add_argument('--epochs',        type=int,   default=100)   # DeiT trains 300, 100 is fair ablation
    p.add_argument('--batch_size',    type=int,   default=128)
    p.add_argument('--lr',            type=float, default=1e-3)  # DeiT / VMamba default
    p.add_argument('--weight_decay',  type=float, default=0.05)  # VMamba / Swin standard
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # TF32 tensor cores

    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nRegularization stack:")
    print(f"  ✓ Mixup (α=0.8) + CutMix (α=1.0) — alternating")
    print(f"  ✓ RandAugment (ops=2, mag=9)")
    print(f"  ✓ Random Erasing (p=0.25)")
    print(f"  ✓ Label Smoothing (ε=0.1)")
    print(f"  ✓ AdamW weight_decay={cfg.weight_decay}")
    print(f"  ✓ Gradient Clipping (max_norm=1.0)")
    print(f"  ✓ Warmup ({5} epochs) + Cosine Annealing LR\n")

    train_loader, val_loader = get_dataloaders(cfg)
    results = {}

    if cfg.mode in ('hilbert', 'all'):
        model_h = HilbertMamba(cfg).to(device)
        model_h = torch.compile(model_h)
        results['HilbertMamba'] = train_model(
            'HilbertMamba', model_h, train_loader, val_loader, cfg, device)

    if cfg.mode in ('spatial', 'all'):
        setattr(cfg, 'canvas_size', cfg.img_size)
        model_s = CSMamba(cfg).to(device)
        model_s = torch.compile(model_s)
        results['CSMamba'] = train_model(
            'CSMamba', model_s, train_loader, val_loader, cfg, device)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  FINAL COMPARISON — Tiny-ImageNet (200 classes)")
        print(f"{'='*60}")
        for name, acc in results.items():
            print(f"  {name:20s}: {acc:.2f}%")
        winner = max(results, key=results.get)
        delta  = max(results.values()) - min(results.values())
        print(f"\n  Winner: {winner} (+{delta:.2f}%)")


if __name__ == '__main__':
    main()
