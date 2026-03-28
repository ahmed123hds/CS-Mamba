"""
Scattered Patch Classification — The Real Routing Test
========================================================
This experiment simulates the WSI problem at small scale.

Setup:
    - Take an MNIST digit (28×28)
    - Embed it into a LARGE canvas (e.g., 112×112 or 224×224)
    - Scatter the digit's patches randomly across the canvas
    - Fill remaining patches with noise (simulating blank tissue/glass)
    
The task: classify the digit from the scattered canvas.

WHY THIS TESTS THE ROUTING HYPOTHESIS:
    - HilbertMamba scans the canvas spatially → the digit's patches
      are scattered across the sequence with 50+ noise patches between them
      → Mamba's state FORGETS the earlier digit patches
    - ODEMamba learns to group the digit patches together in the sequence
      regardless of where they physically sit on the canvas
      → Mamba processes all digit patches contiguously → easy classification

This is exactly what happens in WSIs: tumor patches are scattered
across a sea of healthy tissue.

Usage:
    python train_scattered.py --mode both --epochs 50
    python train_scattered.py --canvas_size 224 --epochs 50  # harder version
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import numpy as np

from models.patch_encoder      import PatchEmbedding
from models.mamba_simple       import MambaClassifier
from models.neural_ode_router  import NeuralODERouter, FixedRouterHilbert
from models.continuous_graph_mamba import ContinuousGraphMambaClassifier as CGMamba
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba


# ──────────────────────────────────────────────────────────────────────
# Scattered MNIST Dataset
# ──────────────────────────────────────────────────────────────────────

class ScatteredMNIST(Dataset):
    """
    Takes MNIST digits and scatters their patches across a large canvas.

    Example with canvas_size=112, patch_size=7:
        - Canvas: 16×16 grid = 256 patches
        - MNIST digit: 4×4 grid = 16 patches (from 28×28 image with 7×7 patches)
        - 16 digit patches placed at random positions in the 256-patch canvas
        - Remaining 240 patches filled with Gaussian noise

    This simulates a WSI where:
        - Digit patches = tumor regions (scattered, important)
        - Noise patches = healthy tissue / glass (filling, irrelevant)
    """

    def __init__(
        self,
        train:       bool = True,
        canvas_size: int  = 112,
        patch_size:  int  = 7,
        noise_std:   float = 0.1,
        seed:        int   = 42,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.patch_size  = patch_size
        self.noise_std   = noise_std
        self.grid_size   = canvas_size // patch_size  # e.g., 16 for 112/7
        self.n_patches   = self.grid_size ** 2        # e.g., 256

        # Load MNIST
        self.mnist = torchvision.datasets.MNIST(
            root='./data', train=train, download=True,
            transform=T.ToTensor()
        )

        # Digit grid: 28 // 7 = 4×4 = 16 patches
        self.digit_grid = 28 // patch_size
        self.n_digit_patches = self.digit_grid ** 2   # 16

        self.rng = np.random.RandomState(seed if not train else seed + 1)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        digit_img, label = self.mnist[idx]   # (1, 28, 28), int
        digit_img = digit_img.squeeze(0)     # (28, 28)

        # Create canvas filled with noise: (1, canvas_size, canvas_size)
        canvas = torch.randn(1, self.canvas_size, self.canvas_size) * self.noise_std

        # Extract digit patches: list of (patch_size, patch_size) tensors
        p = self.patch_size
        digit_patches = []
        for r in range(self.digit_grid):
            for c in range(self.digit_grid):
                patch = digit_img[r*p:(r+1)*p, c*p:(c+1)*p]  # (7, 7)
                digit_patches.append(patch)

        # Pick random positions on the canvas grid for the digit patches
        all_positions = list(range(self.n_patches))
        self.rng.shuffle(all_positions)
        digit_positions = sorted(all_positions[:self.n_digit_patches])

        # Place digit patches on the canvas
        for i, pos in enumerate(digit_positions):
            row = pos // self.grid_size
            col = pos % self.grid_size
            canvas[0, row*p:(row+1)*p, col*p:(col+1)*p] = digit_patches[i]

        # Expand to 3 channels (for compatibility with patch encoder)
        canvas = canvas.expand(3, -1, -1).clone()

        return canvas, label


# ──────────────────────────────────────────────────────────────────────
# Models (same as train.py but with flexible grid/canvas)
# ──────────────────────────────────────────────────────────────────────

class HilbertMambaScattered(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(
            img_size=cfg.canvas_size,
            patch_size=cfg.patch_size,
            in_channels=3,
            d_embed=cfg.d_embed,
        )
        grid = cfg.canvas_size // cfg.patch_size
        self.router = FixedRouterHilbert(grid, grid)
        self.mamba = MambaClassifier(
            d_model=cfg.d_embed,
            n_classes=10,
            n_layers=cfg.n_mamba_layers,
            d_state=cfg.d_state,
        )

    def forward(self, x):
        emb = self.embedder(x)
        emb, _ = self.router(emb)
        return self.mamba(emb)


class ODEMambaScattered(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedder = PatchEmbedding(
            img_size=cfg.canvas_size,
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
        emb = self.embedder(x)
        emb, scores = self.router(emb)
        return self.mamba(emb)


# ──────────────────────────────────────────────────────────────────────
# Training
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


def train_model(name, model, train_loader, val_loader, cfg, device):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Params:   {sum(p.numel() for p in model.parameters()):,}")
    n_patches = (cfg.canvas_size // cfg.patch_size) ** 2
    print(f"  Patches:  {n_patches} per image "
          f"({cfg.canvas_size//cfg.patch_size}×{cfg.canvas_size//cfg.patch_size} grid)")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)

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
            torch.save(model.state_dict(),
                       os.path.join(cfg.save_dir, f"{name}_best.pt"))

        print(f"  Epoch {epoch:03d}/{cfg.epochs} | "
              f"Train {tr_acc:5.1f}% | Val {vl_acc:5.1f}% | "
              f"Time {elapsed:.1f}s | LR {scheduler.get_last_lr()[0]:.2e}")

    print(f"\n  ✓ Best Val Acc: {best_val_acc:.2f}%")
    history['best_val_acc'] = best_val_acc
    return history


def main():
    p = argparse.ArgumentParser(
        description="Scattered MNIST — Routing Hypothesis Test"
    )
    p.add_argument('--mode', choices=['hilbert', 'ode', 'graph_ode', 'spatial', 'all'],
                   default='all')

    # Canvas / patches
    p.add_argument('--canvas_size', type=int, default=112,
                   help="Canvas size (112 → 16×16=256 patches with patch_size=7)")
    p.add_argument('--patch_size',  type=int, default=7)
    p.add_argument('--noise_std',   type=float, default=0.1,
                   help="Std of noise filling non-digit patches")

    # Model
    p.add_argument('--d_embed',        type=int, default=64)
    p.add_argument('--d_state',        type=int, default=16)
    p.add_argument('--n_mamba_layers', type=int, default=2)

    # ODE router
    p.add_argument('--ode_d_ff',   type=int,   default=64)
    p.add_argument('--tau',        type=float, default=0.5,
                   help="NeuralSort temperature (higher = softer)")
    p.add_argument('--ode_solver', default='rk4',
                   choices=['euler', 'rk4', 'dopri5'])
    p.add_argument('--ode_steps',  type=int, default=10)
    p.add_argument('--K_steps',    type=int, default=3,
                   help="Discrete Euler steps for CG-Mamba diffusion")

    # Training
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
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    n_patches = (cfg.canvas_size // cfg.patch_size) ** 2
    print(f"\n  Canvas: {cfg.canvas_size}×{cfg.canvas_size}")
    print(f"  Patch:  {cfg.patch_size}×{cfg.patch_size}")
    print(f"  Grid:   {cfg.canvas_size//cfg.patch_size}×"
          f"{cfg.canvas_size//cfg.patch_size} = {n_patches} patches")
    print(f"  Digit patches: 16 / {n_patches} "
          f"({100*16/n_patches:.1f}% signal, rest is noise)")

    # Data
    train_ds = ScatteredMNIST(
        train=True, canvas_size=cfg.canvas_size,
        patch_size=cfg.patch_size, noise_std=cfg.noise_std, seed=cfg.seed)
    val_ds = ScatteredMNIST(
        train=False, canvas_size=cfg.canvas_size,
        patch_size=cfg.patch_size, noise_std=cfg.noise_std, seed=cfg.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    results = {}

    if cfg.mode in ('hilbert', 'all'):
        model_h = HilbertMambaScattered(cfg).to(device)
        results['HilbertMamba'] = train_model(
            'HilbertMamba', model_h, train_loader, val_loader, cfg, device)

    if cfg.mode in ('ode', 'all'):
        model_o = ODEMambaScattered(cfg).to(device)
        results['ODEMamba'] = train_model(
            'ODEMamba', model_o, train_loader, val_loader, cfg, device)

    if cfg.mode in ('graph_ode', 'all'):
        model_g = CGMamba(cfg).to(device)
        results['CGMamba'] = train_model(
            'CGMamba', model_g, train_loader, val_loader, cfg, device)

    if cfg.mode in ('spatial', 'all'):
        model_s = CSMamba(cfg).to(device)
        results['CSMamba'] = train_model(
            'CSMamba', model_s, train_loader, val_loader, cfg, device)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  FINAL COMPARISON — SCATTERED MNIST")
        print(f"{'='*60}")
        for name, hist in results.items():
            avg_time = np.mean(hist['epoch_time'])
            print(f"  {name:20s}  Best Val: {hist['best_val_acc']:5.2f}%  "
                  f"Avg epoch: {avg_time:.1f}s")

        import json
        with open('results_scattered.json', 'w') as f:
            json.dump({k: {m: v if not isinstance(v, list) else v
                           for m, v in h.items()}
                       for k, h in results.items()}, f, indent=2)
        print("\n  Results saved to results_scattered.json")

    print("\nDone.")


if __name__ == '__main__':
    main()
