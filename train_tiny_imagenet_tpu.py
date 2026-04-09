"""
Tiny-ImageNet V1 vs V2 Ablation — TPU v4-8 Optimized
=====================================================
Adapted from train_tiny_imagenet_v2.py for 8-core TPU training.
Uses HuggingFace datasets (auto-downloaded), XLA ParallelLoader,
and DeiT-level regularization.

Usage (on TPU VM):
  python train_tiny_imagenet_tpu.py --mode v2         # V2 only
  python train_tiny_imagenet_tpu.py --mode v1         # V1 only
  python train_tiny_imagenet_tpu.py --mode compare    # Head-to-head
"""

import os, sys, time, math, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset

# TPU XLA Imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# Model Imports
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba_V1
from models.continuous_spatial_mamba_v2 import CSMamba_V2


# ─────────────────────────────────────────────────────────
#  Regularization: Mixup + CutMix (CPU-safe, no dynamic shapes)
# ─────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, _, H, W = x.size()
    index = torch.randperm(B)
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
#  Dataset (HuggingFace Tiny-ImageNet)
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


# ─────────────────────────────────────────────────────────
#  Training & Evaluation (XLA-optimized)
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, train_loader_raw, train_sampler, optimizer, criterion, device, cfg, epoch):
    model.train()
    train_sampler.set_epoch(epoch)  # Shuffle differently each epoch

    # Fresh ParallelLoader each epoch (it's a one-shot iterator)
    para_loader = pl.ParallelLoader(train_loader_raw, [device])
    loader = para_loader.per_device_loader(device)

    tracker = xm.RateTracker()
    t0 = time.time()
    total_steps = len(train_loader_raw)
    running_correct = torch.tensor(0, dtype=torch.long, device=device)
    running_total = torch.tensor(0, dtype=torch.long, device=device)

    for step, (imgs, labels) in enumerate(loader):
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward()
        xm.optimizer_step(optimizer)
        tracker.add(cfg.batch_size)

        # Accumulate on-device (no .item() sync!)
        running_correct += (outs.argmax(1) == labels).sum()
        running_total += labels.size(0)

        if step % 50 == 0:
            xm.add_step_closure(
                lambda s=step, l=loss: xm.master_print(
                    f"    Step {s}/{total_steps} | Loss: {l.item():.4f} | "
                    f"Rate: {tracker.global_rate():.2f} img/s"
                )
            )

    # Single sync at epoch end
    tr_acc = 100.0 * running_correct.item() / max(running_total.item(), 1)
    return tr_acc, time.time() - t0


@torch.no_grad()
def evaluate(model, val_loader_raw, criterion, device):
    model.eval()

    # Fresh ParallelLoader for eval
    para_loader = pl.ParallelLoader(val_loader_raw, [device])
    loader = para_loader.per_device_loader(device)

    correct_total = torch.tensor(0, dtype=torch.long, device=device)
    count_total = torch.tensor(0, dtype=torch.long, device=device)

    for imgs, labels in loader:
        outs = model(imgs)
        correct_total += (outs.argmax(1) == labels).sum()
        count_total += labels.size(0)

    # Single sync at end
    acc = 100.0 * correct_total.item() / max(count_total.item(), 1)
    return acc


def train_model(name, model, train_loader_raw, train_sampler, val_loader_raw, cfg, device):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    xm.master_print(f"\n{'='*60}")
    xm.master_print(f"  Training: {name}")
    xm.master_print(f"  Params:   {n_params:,}")
    xm.master_print(f"  Dataset:  Tiny-ImageNet (200 classes, 100K images)")
    xm.master_print(f"  Patches:  {(cfg.img_size//cfg.patch_size)**2} per image "
                    f"({cfg.img_size//cfg.patch_size}x{cfg.img_size//cfg.patch_size} grid)")
    xm.master_print(f"  World:    {xm.xrt_world_size()} TPU cores")
    xm.master_print(f"  GlobalBS: {cfg.batch_size * xm.xrt_world_size()}")
    xm.master_print(f"{'='*60}")

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

    best_acc = 0.0
    for ep in range(1, cfg.epochs + 1):
        tr_acc, t_ep = train_one_epoch(
            model, train_loader_raw, train_sampler, optimizer, criterion, device, cfg, ep)

        # Aggregate train acc across TPU cores
        tr_acc_g = xm.mesh_reduce("tr_acc", tr_acc, lambda x: sum(x)/len(x))

        vl_acc = evaluate(model, val_loader_raw, criterion, device)
        vl_acc_g = xm.mesh_reduce("vl_acc", vl_acc, lambda x: sum(x)/len(x))

        scheduler.step()

        if vl_acc_g > best_acc:
            best_acc = vl_acc_g
            if xm.is_master_ordinal():
                xm.save(model.state_dict(), f"best_{name}.pt")

        xm.master_print(
            f"  Epoch {ep:03d}/{cfg.epochs} | "
            f"Train {tr_acc_g:5.1f}% | Val {vl_acc_g:5.1f}% | "
            f"Time {t_ep:4.1f}s | LR {scheduler.get_last_lr()[0]:.2e}"
        )

    xm.master_print(f"\n  \u2713 Best Val Acc: {best_acc:.2f}%\n")
    return best_acc


# ─────────────────────────────────────────────────────────
#  XLA Worker Function
# ─────────────────────────────────────────────────────────
def _mp_fn(index, flags):
    device = xm.xla_device()
    torch.manual_seed(flags.seed + index)
    np.random.seed(flags.seed + index)
    random.seed(flags.seed + index)

    xm.master_print(f"\n{'='*60}")
    xm.master_print(f"  CS-Mamba V1 vs V2 — Tiny-ImageNet-200 Ablation (TPU)")
    xm.master_print(f"{'='*60}")
    xm.master_print(f"  V1: Collapsed diffusion (sum over S, manual slicing)")
    xm.master_print(f"  V2: State-preserving diffusion (per-S, F.conv2d)")
    xm.master_print(f"{'='*60}\n")

    # ── Dataset: each spawned process reads from HF cache independently ──
    xm.master_print("  [Loading Tiny-ImageNet from HuggingFace cache...]")
    ds = load_dataset("Maysee/tiny-imagenet")

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(flags.img_size, scale=(0.67, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # NOTE: RandomErasing removed — dynamic mask shapes cause XLA recompilation
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = TinyImageNetDataset(ds['train'], transform_train)
    val_ds   = TinyImageNetDataset(ds['valid'], transform_val)

    # ── Distributed Samplers for XLA ─────────────────────
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    train_loader_raw = DataLoader(
        train_ds, batch_size=flags.batch_size,
        sampler=train_sampler, num_workers=0,
        drop_last=True
    )
    val_loader_raw = DataLoader(
        val_ds, batch_size=flags.batch_size,
        sampler=val_sampler, num_workers=0,
        drop_last=False
    )

    # Wrap with XLA ParallelLoader for async device transfer
    # NOTE: Do NOT wrap here — ParallelLoader is a one-shot iterator.
    # train_model() creates a fresh ParallelLoader each epoch.

    # ── Config ───────────────────────────────────────────
    setattr(flags, 'canvas_size', flags.img_size)

    results = {}

    if flags.mode in ('v1', 'compare'):
        model_v1 = CSMamba_V1(flags).to(device)
        results['CSMamba_V1'] = train_model(
            'CSMamba_V1', model_v1, train_loader_raw, train_sampler, val_loader_raw, flags, device)
        del model_v1
        xm.mark_step()

    if flags.mode in ('v2', 'compare'):
        model_v2 = CSMamba_V2(flags).to(device)
        results['CSMamba_V2'] = train_model(
            'CSMamba_V2', model_v2, train_loader_raw, train_sampler, val_loader_raw, flags, device)
        del model_v2
        xm.mark_step()

    if len(results) > 1:
        xm.master_print(f"\n{'='*60}")
        xm.master_print(f"  FINAL COMPARISON — Tiny-ImageNet-200")
        xm.master_print(f"{'='*60}")
        for name, acc in results.items():
            xm.master_print(f"  {name:20s}: {acc:.2f}%")
        winner = max(results, key=results.get)
        delta = max(results.values()) - min(results.values())
        xm.master_print(f"\n  Winner: {winner} (+{delta:.2f}%)")


# ─────────────────────────────────────────────────────────
#  Args & Entry
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("CS-Mamba V1 vs V2 — Tiny-ImageNet (TPU)")
    p.add_argument('--mode',           choices=['v1', 'v2', 'compare'], default='v2')
    p.add_argument('--img_size',       type=int,   default=64)
    p.add_argument('--patch_size',     type=int,   default=4)
    p.add_argument('--n_classes',      type=int,   default=200)
    p.add_argument('--d_embed',        type=int,   default=192)
    p.add_argument('--d_state',        type=int,   default=16)
    p.add_argument('--n_mamba_layers', type=int,   default=8)
    p.add_argument('--K_steps',        type=int,   default=3)
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch_size',     type=int,   default=128, help="Per TPU core")
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=0.05)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    flags = parse_args()

    # Pre-download dataset in main process so all spawned workers hit the local cache
    print("[Main] Pre-fetching Tiny-ImageNet to local HuggingFace cache...")
    load_dataset("Maysee/tiny-imagenet")
    print("[Main] Dataset cached. Spawning TPU workers with start_method='spawn'...")

    # Use 'fork' — inherits parent's /dev/accel0 TPU device permissions
    # Original fork issues (deadlock, slow training) were caused by:
    #   1. HuggingFace lock contention (fixed: pre-cached in main)
    #   2. CutMix dynamic shapes (fixed: removed from training loop)
    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='fork')

