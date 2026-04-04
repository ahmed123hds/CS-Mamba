"""
Tiny-ImageNet Training Script for CS-Mamba V2
==============================================
Tests the fixed State-Preserving Diffusion architecture against the
original collapsed-diffusion V1 on Tiny-ImageNet-200.

Run:
  python train_tiny_imagenet_v2.py --mode v2         # V2 only
  python train_tiny_imagenet_v2.py --mode v1         # V1 only (baseline)
  python train_tiny_imagenet_v2.py --mode compare    # Head-to-head V1 vs V2
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
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba_V1
from models.continuous_spatial_mamba_v2 import CSMamba_V2


# ─────────────────────────────────────────────────────────
#  Regularization: Mixup + CutMix
# ─────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.8, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    lam = np.random.beta(alpha, alpha)
    B, _, H, W = x.size()
    index = torch.randperm(B).to(device)
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
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

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.67, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = TinyImageNetDataset(ds['train'], transform_train)
    val_ds   = TinyImageNetDataset(ds['valid'], transform_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────
#  Train One Epoch
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, cfg):
    model.train()
    correct = total = total_loss = 0
    t0 = time.time()
    use_mixup = random.random() < 0.5

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.8, device=device)
        else:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=1.0, device=device)

        with autocast('cuda'):
            outs = model(imgs)
            loss = mixup_criterion(criterion, outs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

        if step % 50 == 0:
            print(f"    Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / total, 100.0 * correct / total, time.time() - t0


# ─────────────────────────────────────────────────────────
#  Validate
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
    print(f"  Patches:  {(cfg.img_size//cfg.patch_size)**2} per image "
          f"({cfg.img_size//cfg.patch_size}x{cfg.img_size//cfg.patch_size} grid)")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.999))

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
    p = argparse.ArgumentParser("CS-Mamba V1 vs V2 Comparison on Tiny-ImageNet")
    p.add_argument('--mode',           choices=['v1', 'v2', 'compare'], default='v2')
    p.add_argument('--img_size',       type=int,   default=64)
    p.add_argument('--patch_size',     type=int,   default=4)
    p.add_argument('--n_classes',      type=int,   default=200)
    p.add_argument('--d_embed',        type=int,   default=192)
    p.add_argument('--d_state',        type=int,   default=16)
    p.add_argument('--n_mamba_layers', type=int,   default=8)
    p.add_argument('--K_steps',        type=int,   default=3)
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch_size',     type=int,   default=128)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=0.05)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n{'='*60}")
    print(f"  CS-Mamba V1 vs V2 — Tiny-ImageNet-200 Benchmark")
    print(f"{'='*60}")
    print(f"  V1: Collapsed diffusion (sum over S, manual slicing)")
    print(f"  V2: State-preserving diffusion (per-S, F.conv2d)")
    print(f"{'='*60}\n")

    setattr(cfg, 'canvas_size', cfg.img_size)
    train_loader, val_loader = get_dataloaders(cfg)
    results = {}

    if cfg.mode in ('v1', 'compare'):
        model_v1 = CSMamba_V1(cfg).to(device)
        # model_v1 = torch.compile(model_v1)  # Skip on consumer GPUs (30min+ overhead)
        results['CSMamba_V1'] = train_model(
            'CSMamba_V1', model_v1, train_loader, val_loader, cfg, device)

    if cfg.mode in ('v2', 'compare'):
        model_v2 = CSMamba_V2(cfg).to(device)
        # model_v2 = torch.compile(model_v2)  # Skip on consumer GPUs (30min+ overhead)
        results['CSMamba_V2'] = train_model(
            'CSMamba_V2', model_v2, train_loader, val_loader, cfg, device)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  FINAL COMPARISON — Tiny-ImageNet-200")
        print(f"{'='*60}")
        for name, acc in results.items():
            print(f"  {name:20s}: {acc:.2f}%")
        winner = max(results, key=results.get)
        delta = max(results.values()) - min(results.values())
        print(f"\n  Winner: {winner} (+{delta:.2f}%)")


if __name__ == '__main__':
    main()
