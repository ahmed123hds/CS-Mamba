"""
CS-Mamba V4 — ImageNet-1K Production Training Script (TPU v4)
==============================================================
Gold-standard DeiT/VMamba training recipe for ICLR 2026 submission.

Training Recipe (matches DeiT / Vim / VMamba exactly):
  Optimizer:      AdamW (β1=0.9, β2=0.999)
  Base LR:        1e-3  (linear scaled: lr = 1e-3 × GlobalBS/1024)
  Weight Decay:   0.05
  Warmup:         20 epochs (linear from 1e-6 → peak)
  Schedule:       Cosine Annealing → eta_min=1e-5
  Epochs:         300
  Grad Clip:      max_norm=1.0
  Label Smoothing: 0.1
  MixUp α:        0.8
  CutMix α:       1.0
  RandAugment:    (2, 9)
  Drop Path:      0.1 (Small), 0.2 (Base)

Usage (v4-8, single host):
  PJRT_DEVICE=TPU python train_tpu_wds_v4.py \
    --train_shards 'gs://bucket/imagenet1k-train-{0000..1023}.tar' \
    --val_shards   'gs://bucket/imagenet1k-val-{00..63}.tar'

Usage (v4-32, multi-host pod):
  python train_tpu_wds_v4.py --train_shards '...' --val_shards '...' --multi_host
"""

import os
import sys
import time
import math
import argparse
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds

# TPU XLA
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# CS-Mamba V4
from models.continuous_spatial_mamba_v4 import CSMamba_V4


# ════════════════════════════════════════════════════════════════════
# Hyperparameter Defaults (Gold-Standard DeiT/VMamba Recipe)
# ════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser("CS-Mamba V4 — ImageNet-1K (TPU)")

    # ── Dataset URLs ──
    p.add_argument('--train_shards', type=str, required=True)
    p.add_argument('--val_shards',   type=str, required=True)

    # ── Model Architecture ──
    p.add_argument('--img_size',       type=int,   default=224)
    p.add_argument('--patch_size',     type=int,   default=16)
    p.add_argument('--d_embed',        type=int,   default=384,   help="384=Small, 768=Base")
    p.add_argument('--n_mamba_layers', type=int,   default=12,    help="12=Small, 24=Base")
    p.add_argument('--K_steps',        type=int,   default=3)
    p.add_argument('--n_classes',      type=int,   default=1000)
    p.add_argument('--drop_path',      type=float, default=0.1,   help="0.1=Small, 0.2=Base")

    # ── Optimizer (DeiT/VMamba standard) ──
    p.add_argument('--batch_size',   type=int,   default=64,    help="Per TPU core")
    p.add_argument('--epochs',       type=int,   default=300)
    p.add_argument('--base_lr',      type=float, default=1e-3,  help="Base LR for GlobalBS=1024")
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--grad_clip',    type=float, default=1.0,   help="Max gradient norm")
    p.add_argument('--warmup_epochs', type=int,  default=20)
    p.add_argument('--min_lr',       type=float, default=1e-5,  help="Cosine annealing floor")

    # ── Regularization ──
    p.add_argument('--mixup_alpha',  type=float, default=0.8)
    p.add_argument('--cutmix_alpha', type=float, default=1.0)
    p.add_argument('--mixup_prob',   type=float, default=0.5,   help="Prob of MixUp vs CutMix")
    p.add_argument('--label_smooth', type=float, default=0.1)

    # ── Checkpoint ──
    p.add_argument('--resume',       type=str,   default='',    help="Checkpoint .pt to resume")
    p.add_argument('--save_dir',     type=str,   default='.',   help="Where to save checkpoints")
    p.add_argument('--save_every',   type=int,   default=10,    help="Save checkpoint every N epochs")

    # ── Infrastructure ──
    p.add_argument('--multi_host',   action='store_true')
    p.add_argument('--num_workers',  type=int,   default=8)
    p.add_argument('--seed',         type=int,   default=42)

    return p.parse_args()


class EmptyConfig:
    pass


# ════════════════════════════════════════════════════════════════════
#  MixUp & CutMix (Pure PyTorch, XLA-safe — no dynamic shapes)
# ════════════════════════════════════════════════════════════════════
def mixup_data(images, labels, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0))
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


def cutmix_data(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, _, H, W = images.shape
    index = torch.randperm(B)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    return mixed, labels, labels[index], lam


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ════════════════════════════════════════════════════════════════════
#  Cosine Schedule with Linear Warmup
# ════════════════════════════════════════════════════════════════════
def build_lr_scheduler(optimizer, flags, scaled_lr):
    """DeiT / VMamba standard: 20-epoch linear warmup → cosine → min_lr."""
    warmup_epochs = flags.warmup_epochs
    total_epochs = flags.epochs
    min_lr = flags.min_lr

    def lr_lambda(epoch):
        # Linear warmup phase
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Cosine annealing phase
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale between min_lr and peak_lr
        return max(min_lr / scaled_lr, cosine_factor)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════
#  WebDataset Loader (GCS streaming)
# ════════════════════════════════════════════════════════════════════
def build_wds_loader(shards_url, batch_size, flags, is_training=True):
    if is_training:
        transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.08, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def apply_transforms(sample):
        image, label = sample
        return transform(image), label

    def apply_mixup_cutmix(batch):
        images, labels = batch
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = labels.long()
        if np.random.random() < flags.mixup_prob:
            mixed, la, lb, lam = mixup_data(images, labels, flags.mixup_alpha)
        else:
            mixed, la, lb, lam = cutmix_data(images, labels, flags.cutmix_alpha)
        return mixed, la, lb, torch.tensor(lam, dtype=torch.float32)

    def apply_stack_val(batch):
        images, labels = batch
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = labels.long()
        return images, labels

    nodesplitter = wds.split_by_node if flags.multi_host else wds.split_by_worker

    dataset = (
        wds.WebDataset(shards_url, resampled=True, nodesplitter=nodesplitter)
        .shuffle(5000 if is_training else 0)
        .decode("pil")
        .to_tuple("jpg;png", "cls")
        .map(apply_transforms)
        .batched(batch_size, partial=False)
    )

    if is_training:
        dataset = dataset.map(apply_mixup_cutmix)
    else:
        dataset = dataset.map(apply_stack_val)

    loader = wds.WebLoader(
        dataset, batch_size=None,
        num_workers=flags.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    return loader


# ════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ════════════════════════════════════════════════════════════════════
def _mp_fn(index, flags):
    device = xm.xla_device()
    torch.manual_seed(flags.seed + index)
    np.random.seed(flags.seed + index)
    random.seed(flags.seed + index)

    # ── Model Config ──
    cfg = EmptyConfig()
    cfg.img_size       = flags.img_size
    cfg.patch_size     = flags.patch_size
    cfg.d_embed        = flags.d_embed
    cfg.n_mamba_layers = flags.n_mamba_layers
    cfg.K_steps        = flags.K_steps
    cfg.n_classes      = flags.n_classes
    cfg.canvas_size    = flags.img_size
    cfg.drop_path      = flags.drop_path

    # ── Build Model ──
    model = CSMamba_V4(cfg).to(device)

    # ── Linear LR Scaling Rule ──
    world_size = xm.xrt_world_size()
    global_bs = flags.batch_size * world_size
    scaled_lr = flags.base_lr * (global_bs / 1024.0)

    # ── Optimizer: AdamW with correct WD ──
    criterion = nn.CrossEntropyLoss(label_smoothing=flags.label_smooth)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=flags.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── LR Scheduler: Warmup + Cosine ──
    scheduler = build_lr_scheduler(optimizer, flags, scaled_lr)

    start_epoch = 1
    best_acc = 0.0

    # ── Resume ──
    if flags.resume:
        xm.master_print(f"📦 Resuming from: {flags.resume}")
        ckpt = torch.load(flags.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        xm.master_print(f"✅ Resumed from epoch {start_epoch}, best={best_acc:.2f}%")

    # ── Data Loaders ──
    train_loader = build_wds_loader(flags.train_shards, flags.batch_size, flags, is_training=True)
    val_loader   = build_wds_loader(flags.val_shards,   flags.batch_size, flags, is_training=False)

    train_steps = math.ceil(1_281_167 / global_bs)
    val_steps   = math.ceil(50_000   / global_bs)

    para_train = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    para_val   = pl.ParallelLoader(val_loader,   [device]).per_device_loader(device)

    n_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"\n{'='*70}")
    xm.master_print(f"  CS-Mamba V4 — ImageNet-1K Training (ICLR 2026)")
    xm.master_print(f"{'='*70}")
    xm.master_print(f"  Params:        {n_params:,} ({n_params/1e6:.1f}M)")
    xm.master_print(f"  Architecture:  d={flags.d_embed}, layers={flags.n_mamba_layers}, "
                    f"K={flags.K_steps}, drop_path={flags.drop_path}")
    xm.master_print(f"  World Size:    {world_size} TPU cores")
    xm.master_print(f"  Global BS:     {global_bs}")
    xm.master_print(f"  Steps/Epoch:   {train_steps} train, {val_steps} val")
    xm.master_print(f"  Scaled LR:     {scaled_lr:.6f} (base={flags.base_lr} × {global_bs}/1024)")
    xm.master_print(f"  Weight Decay:  {flags.weight_decay}")
    xm.master_print(f"  Warmup:        {flags.warmup_epochs} epochs → Cosine → {flags.min_lr}")
    xm.master_print(f"  Grad Clip:     max_norm={flags.grad_clip}")
    xm.master_print(f"  Augmentation:  RandAug(2,9) + ColorJitter(0.4) + "
                    f"MixUp({flags.mixup_alpha}) + CutMix({flags.cutmix_alpha})")
    xm.master_print(f"  Label Smooth:  {flags.label_smooth}")
    xm.master_print(f"{'='*70}\n")

    # ╔══════════════════════════════════════════════════════════════╗
    # ║                    TRAINING LOOP                           ║
    # ╚══════════════════════════════════════════════════════════════╝
    for epoch in range(start_epoch, flags.epochs + 1):
        # ── TRAIN ──
        model.train()
        tracker = xm.RateTracker()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for step, batch in enumerate(para_train):
            if step >= train_steps:
                break

            images, labels_a, labels_b, lam = batch

            optimizer.zero_grad()
            logits = model(images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss.backward()

            # ── Gradient Clipping (prevents PDE gradient spikes) ──
            xm.reduce_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), flags.grad_clip)

            xm.optimizer_step(optimizer)
            tracker.add(flags.batch_size)

            total_loss += loss.item() * images.size(0)
            correct += logits.argmax(1).eq(labels_a).sum().item()
            total += images.size(0)

            if step % 100 == 0:
                xm.master_print(
                    f"  E{epoch:03d} | Step {step:4d}/{train_steps} | "
                    f"Loss: {loss.item():.4f} | Rate: {tracker.global_rate():.1f} img/s"
                )

        train_time = time.time() - t0
        g_loss = xm.mesh_reduce("tr_loss", total_loss, np.sum) / max(xm.mesh_reduce("tr_n", total, np.sum), 1)
        g_acc  = 100.0 * xm.mesh_reduce("tr_c", correct, np.sum) / max(xm.mesh_reduce("tr_n", total, np.sum), 1)

        # ── VALIDATE ──
        model.eval()
        v_correct, v_total = 0, 0

        for step, batch in enumerate(para_val):
            if step >= val_steps:
                break
            images, labels = batch
            with torch.no_grad():
                logits = model(images)
            v_correct += logits.argmax(1).eq(labels).sum().item()
            v_total += images.size(0)

        v_acc = 100.0 * xm.mesh_reduce("v_c", v_correct, np.sum) / max(xm.mesh_reduce("v_n", v_total, np.sum), 1)

        # ── LR Step ──
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        # ── Log ──
        xm.master_print(
            f"\n  ═══ Epoch {epoch:03d}/{flags.epochs} | "
            f"Train {g_acc:.1f}% (loss {g_loss:.4f}) | "
            f"Val {v_acc:.1f}% | "
            f"Time {train_time:.0f}s | LR {curr_lr:.2e} ═══\n"
        )

        # ── Save Best ──
        if v_acc > best_acc:
            best_acc = v_acc
            if xm.is_master_ordinal():
                xm.save(model.state_dict(),
                        os.path.join(flags.save_dir, "csmamba_v4_best.pt"))
                xm.master_print(f"  💾 New best! Val Acc: {best_acc:.2f}%")

        # ── Periodic Checkpoint (full state for resumption) ──
        if xm.is_master_ordinal() and (epoch % flags.save_every == 0 or epoch == flags.epochs):
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': v_acc,
                'best_acc': best_acc,
            }
            xm.save(state, os.path.join(flags.save_dir, f"csmamba_v4_ep{epoch}.pt"))
            xm.master_print(f"  📦 Checkpoint saved: epoch {epoch}")

    xm.master_print(f"\n🏁 Training Complete! Best Val Acc: {best_acc:.2f}%")


if __name__ == '__main__':
    flags = parse_args()
    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='fork')
