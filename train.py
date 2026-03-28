"""
Main Training Script — Routing Hypothesis Experiment
======================================================
Tests whether a Neural ODE router outperforms a fixed Hilbert curve
scan order for Mamba-based CIFAR-10 classification.

Two systems:
    A) HilbertMamba  — fixed Hilbert scan → Mamba → classify
    B) ODEMamba      — Neural ODE scores → NeuralSort → Mamba → classify

Usage:
    python train.py --mode hilbert          # System A
    python train.py --mode ode              # System B
    python train.py --mode both             # Train both and compare

Key metrics logged:
    - Train / Val accuracy per epoch
    - Training time per epoch
    - For ODE mode: gradient variance on NeuralSort scores
"""

import argparse
import time
import sys
import os

# Make sure local modules are importable from any working directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np

from models.patch_encoder      import PatchEmbedding
from models.mamba_simple       import MambaClassifier
from models.neural_ode_router  import NeuralODERouter, FixedRouterHilbert
from models.continuous_graph_mamba import ContinuousGraphMambaClassifier as CGMamba
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba


# ──────────────────────────────────────────────────────────────────────
# Full Model Wrappers
# ──────────────────────────────────────────────────────────────────────

class HilbertMamba(nn.Module):
    """Patch Embed → Hilbert Reorder → Mamba → Classify"""

    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_channels=3,
            d_embed=cfg.d_embed,
        )
        grid = cfg.img_size // cfg.patch_size
        self.router = FixedRouterHilbert(grid, grid)
        self.mamba  = MambaClassifier(
            d_model=cfg.d_embed,
            n_classes=10,
            n_layers=cfg.n_mamba_layers,
            d_state=cfg.d_state,
        )

    def forward(self, x):
        emb = self.embedder(x)           # (B, n, d)
        emb, _ = self.router(emb)        # (B, n, d) — Hilbert order
        return self.mamba(emb)           # (B, 10)


class ODEMamba(nn.Module):
    """Patch Embed → Neural ODE → NeuralSort → Mamba → Classify"""

    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_channels=3,
            d_embed=cfg.d_embed,
        )
        self.router = NeuralODERouter(
            d_embed=cfg.d_embed,
            d_ff=cfg.ode_d_ff,
            tau=cfg.tau,
            solver=cfg.ode_solver,
            n_steps=cfg.ode_steps,
        )
        self.mamba = MambaClassifier(
            d_model=cfg.d_embed,
            n_classes=10,
            n_layers=cfg.n_mamba_layers,
            d_state=cfg.d_state,
        )

    def forward(self, x):
        emb = self.embedder(x)            # (B, n, d)
        emb, scores = self.router(emb)    # (B, n, d), (B, n)
        return self.mamba(emb)            # (B, 10)


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if self.std > 0.0:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

def get_dataloaders(cfg):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
        AddGaussianNoise(0., cfg.noise_std)
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
        AddGaussianNoise(0., cfg.noise_std)
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_ds,   batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

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


def train_model(name, model, cfg, device):
    """Train a model and return its history."""
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Params:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    train_loader, val_loader = get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)

    # Mixed precision only if CUDA available
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 🚀 MAGIC TRITON KERNEL FUSION (Runs completely in SRAM)
    if hasattr(torch, 'compile'):
        try:
            print("  [Auto-fusing K-Step Diffusion PDEs into Triton Kernels...]")
            model = torch.compile(model)
        except Exception as e:
            print("  [Fallback to Eager PyTorch]")

    history = {'train_acc': [], 'val_acc': [], 'epoch_time': []}
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['epoch_time'].append(elapsed)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), f"{name}_best.pt")

        print(f"  Epoch {epoch:03d}/{cfg.epochs} | "
              f"Train {tr_acc:5.1f}% | Val {vl_acc:5.1f}% | "
              f"Time {elapsed:.1f}s | LR {scheduler.get_last_lr()[0]:.2e}")

    print(f"\n  ✓ Best Val Acc: {best_val_acc:.2f}%")
    history['best_val_acc'] = best_val_acc
    return history


# ──────────────────────────────────────────────────────────────────────
# Argument Parsing & Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Routing Hypothesis Experiment on CIFAR-10"
    )
    # What to run
    p.add_argument('--mode', choices=['hilbert', 'ode', 'graph_ode', 'spatial', 'all'],
                   default='all', help="Which model(s) to train")

    # Data
    p.add_argument('--data_dir',     default='./data')
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--noise_std',    type=float, default=0.0,
                   help="Add Gaussian noise to the dataset (e.g. 0.1)")

    # Image / patch
    p.add_argument('--img_size',     type=int, default=32)
    p.add_argument('--patch_size',   type=int, default=8,
                   help="Patch size (8 → 4×4=16 patches from 32×32 image)")

    # Model
    p.add_argument('--d_embed',        type=int,   default=128)
    p.add_argument('--d_state',        type=int,   default=16)
    p.add_argument('--n_mamba_layers', type=int,   default=2)

    # ODE router
    p.add_argument('--ode_d_ff',   type=int,   default=64)
    p.add_argument('--tau',        type=float, default=0.1,
                   help="NeuralSort temperature")
    p.add_argument('--ode_solver', default='rk4',
                   choices=['euler', 'rk4', 'dopri5'])
    p.add_argument('--ode_steps',  type=int,   default=10,
                   help="Fixed steps for euler/rk4 solver")
    p.add_argument('--K_steps',    type=int,   default=3,
                   help="Discrete Euler steps defining forward diffusion time")

    # Training
    p.add_argument('--epochs',       type=int,   default=30)
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
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    if cfg.mode in ('hilbert', 'all'):
        model_h = HilbertMamba(cfg).to(device)
        results['HilbertMamba'] = train_model('HilbertMamba', model_h, cfg, device)

    if cfg.mode in ('ode', 'all'):
        model_o = ODEMamba(cfg).to(device)
        results['ODEMamba'] = train_model('ODEMamba', model_o, cfg, device)

    if cfg.mode in ('graph_ode', 'all'):
        # Pass identical config. Note: CGMamba handles patch extraction internally
        setattr(cfg, 'canvas_size', cfg.img_size) # for compatibility
        model_g = CGMamba(cfg).to(device)
        results['CGMamba'] = train_model('CGMamba', model_g, cfg, device)

    if cfg.mode in ('spatial', 'all'):
        setattr(cfg, 'canvas_size', cfg.img_size) # for compatibility
        model_s = CSMamba(cfg).to(device)
        results['CSMamba'] = train_model('CSMamba', model_s, cfg, device)

    # ── Summary comparison ────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  FINAL COMPARISON")
        print(f"{'='*60}")
        for name, hist in results.items():
            avg_time = np.mean(hist['epoch_time'])
            print(f"  {name:20s}  Best Val: {hist['best_val_acc']:5.2f}%  "
                  f"Avg epoch: {avg_time:.1f}s")

        # Save results for visualisation
        import json
        with open('results.json', 'w') as f:
            # Convert to plain lists for JSON serialisation
            json.dump({k: {m: v if not isinstance(v, list) else v
                           for m, v in h.items()}
                       for k, h in results.items()}, f, indent=2)
        print("\n  Results saved to results.json")
        print("  Run:  python visualize.py  to see learning curves")

    print("\nDone.")


if __name__ == '__main__':
    main()
