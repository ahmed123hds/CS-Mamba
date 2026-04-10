"""
TPU V4-32 WebDataset Streaming Script for CS-Mamba V2 (State-Preserving)
========================================================================
Multi-host Pod training with V2 architecture.
DeiT-level Regularization: MixUp, CutMix, RandAugment, Label Smoothing.
Supports checkpoint resumption via --resume flag.

Usage (single-host v4-8):
  python train_tpu_wds_v2.py --train_shards '...' --val_shards '...'

Usage (multi-host v4-32, run on EACH host):
  python train_tpu_wds_v2.py --train_shards '...' --val_shards '...' --multi_host
"""

import os
import sys

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

import time
import argparse
from math import ceil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds

# TPU XLA Imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# ── V3 Model Import (Reaction-Diffusion) ──
from models.continuous_spatial_mamba_v3 import CSMamba_V3 as CSMamba_V2

def parse_args():
    p = argparse.ArgumentParser()
    
    # ── Dataset URLs ──
    p.add_argument('--train_shards', type=str, required=True,
                   help='e.g., /path/to/imagenet1k-train-{0000..1023}.tar')
    p.add_argument('--val_shards', type=str, required=True,
                   help='e.g., /path/to/imagenet1k-validation-{00..63}.tar')
                   
    # ── Model Geometry ──
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--d_embed', type=int, default=384)
    p.add_argument('--d_state', type=int, default=16)
    p.add_argument('--n_mamba_layers', type=int, default=12)
    p.add_argument('--K_steps', type=int, default=3)
    p.add_argument('--n_classes', type=int, default=1000)
    
    # ── Training Params ──
    p.add_argument('--batch_size', type=int, default=64, help="Batch size PER TPU CORE")
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    
    # ── Checkpoint Resume ──
    p.add_argument('--resume', type=str, default='', help="Path to checkpoint .pt file to resume from")
    
    # ── Multi-Host (Pod) ──
    p.add_argument('--multi_host', action='store_true', help="Enable multi-host Pod training")
    p.add_argument('--num_workers', type=int, default=8, help="Dataloader workers per process (increase to fix I/O bottleneck)")
    
    # ── Regularization (DeiT-level) ──
    p.add_argument('--mixup_alpha', type=float, default=0.8)
    p.add_argument('--cutmix_alpha', type=float, default=1.0)
    p.add_argument('--mixup_prob', type=float, default=0.5)
    
    return p.parse_args()


class EmptyConfig:
    pass


# ════════════════════════════════════════════════════════════════════
# MixUp & CutMix Implementation (Pure PyTorch, CPU-Safe)
# ════════════════════════════════════════════════════════════════════

def mixup_data(images, labels, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def cutmix_data(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    
    _, _, H, W = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    return mixed_images, labels, labels[index], lam


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ════════════════════════════════════════════════════════════════════

def build_wds_loader(shards_url, batch_size, flags=None, is_training=True):
    if is_training:
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9), 
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def apply_transforms(sample):
        image, label = sample
        return transform(image), label
        
    def apply_mixup_cutmix_batched(batch):
        images, labels = batch
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = labels.long()
        
        use_mixup = np.random.random() < flags.mixup_prob
        if use_mixup:
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, flags.mixup_alpha)
        else:
            mixed_images, labels_a, labels_b, lam = cutmix_data(images, labels, flags.cutmix_alpha)
            
        lam_tensor = torch.tensor(lam, dtype=torch.float32)
        return mixed_images, labels_a, labels_b, lam_tensor

    def apply_stack_val(batch):
        images, labels = batch
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = labels.long()
        return images, labels

    # Use split_by_node for multi-host Pod training
    nodesplitter = wds.split_by_node if flags and flags.multi_host else wds.split_by_worker

    num_workers = flags.num_workers if flags else 8

    dataset = (
        wds.WebDataset(shards_url, resampled=True, nodesplitter=nodesplitter)
        .shuffle(2000 if is_training else 0)
        .decode("pil")
        .to_tuple("jpg;png", "cls")
        .map(apply_transforms)
        .batched(batch_size, partial=False)
    )

    if is_training:
        dataset = dataset.map(apply_mixup_cutmix_batched)
    else:
        dataset = dataset.map(apply_stack_val)

    loader = wds.WebLoader(
        dataset, batch_size=None,
        num_workers=num_workers,  # More workers to saturate GCS bandwidth
        pin_memory=True,           # Faster host→device transfer
        prefetch_factor=4,         # Each worker prefetches 4 batches ahead
        persistent_workers=True,   # Avoid worker respawn overhead
    )
    return loader


def _mp_fn(index, flags):
    device = xm.xla_device()
    
    # ── Model Config ─────────────────────────────────────────
    cfg = EmptyConfig()
    cfg.img_size = flags.img_size
    cfg.patch_size = flags.patch_size
    cfg.d_embed = flags.d_embed
    cfg.d_state = flags.d_state
    cfg.n_mamba_layers = flags.n_mamba_layers
    cfg.K_steps = flags.K_steps
    cfg.n_classes = flags.n_classes
    cfg.canvas_size = flags.img_size

    # ── Init V2 Model ────────────────────────────────────────
    model = CSMamba_V2(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=flags.epochs, eta_min=1e-6)
    
    start_epoch = 1
    
    # ── Resume from Checkpoint ───────────────────────────────
    if flags.resume:
        xm.master_print(f"📦 Resuming from checkpoint: {flags.resume}")
        ckpt = torch.load(flags.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        xm.master_print(f"✅ Resumed! Starting from epoch {start_epoch} | "
                        f"Previous Val Acc: {ckpt.get('val_acc', 'N/A')}")
    
    # ── Loaders ──────────────────────────────────────────────
    train_loader = build_wds_loader(flags.train_shards, flags.batch_size, flags, is_training=True)
    val_loader   = build_wds_loader(flags.val_shards,   flags.batch_size, flags, is_training=False)
    
    world_size = xm.xrt_world_size()
    global_batch_size = flags.batch_size * world_size
    train_steps = ceil(1281167 / global_batch_size)
    val_steps   = ceil(50000   / global_batch_size)
    
    para_train = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    para_val   = pl.ParallelLoader(val_loader,   [device]).per_device_loader(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"\n🚀 Launching CS-Mamba V2 (State-Preserving Diffusion)")
    xm.master_print(f"   Params: {n_params:,} | World Size: {world_size} cores")
    xm.master_print(f"   Global Batch Size: {global_batch_size} | "
                    f"Train Steps/Epoch: {train_steps} | Val Steps: {val_steps}")
    xm.master_print(f"   d_embed={flags.d_embed} | n_layers={flags.n_mamba_layers} | "
                    f"K_steps={flags.K_steps} | d_state={flags.d_state}")
    xm.master_print(f"   🛡️  MixUp(α={flags.mixup_alpha}) | CutMix(α={flags.cutmix_alpha}) | "
                    f"RandAugment | Label Smoothing=0.1 | WD={flags.weight_decay}")
    
    best_acc = 0.0
    
    for epoch in range(start_epoch, flags.epochs + 1):
        
        # ── 1. TRAINING PHASE ────────────────────────────────────
        model.train()
        tracker = xm.RateTracker()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        
        for step, batch in enumerate(para_train):
            if step >= train_steps: break
            
            images, labels_a, labels_b, lam = batch
                
            optimizer.zero_grad()
            logits = model(images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(flags.batch_size)
            
            total_loss += loss.item() * images.size(0)
            correct += logits.argmax(dim=1).eq(labels_a).sum().item()
            total += images.size(0)
            
            if step % 50 == 0:
                xm.master_print(
                    f"Epoch {epoch} | Step {step}/{train_steps} | Loss: {loss.item():.4f} | "
                    f"Rate: {tracker.global_rate():.2f} img/s"
                )

        global_loss    = xm.mesh_reduce("tr_loss", total_loss, np.sum) / xm.mesh_reduce("tr_tot", total, np.sum)
        global_correct = 100.0 * xm.mesh_reduce("tr_corr", correct, np.sum) / xm.mesh_reduce("tr_tot", total, np.sum)
        train_time = time.time() - t0
        
        # ── 2. VALIDATION PHASE ──────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        for step, batch in enumerate(para_val):
            if step >= val_steps: break
            
            images, labels = batch
            
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)
                
            val_loss += loss.item() * images.size(0)
            val_correct += logits.argmax(dim=1).eq(labels).sum().item()
            val_total += images.size(0)

        global_v_loss = xm.mesh_reduce("v_loss", val_loss, np.sum) / xm.mesh_reduce("v_tot", val_total, np.sum)
        global_v_acc  = 100.0 * xm.mesh_reduce("v_corr", val_correct, np.sum) / xm.mesh_reduce("v_tot", val_total, np.sum)
        
        # ── 3. LOGGING & SAVING ──────────────────────────────────
        xm.master_print(
            f"\n==== Epoch {epoch} Completed in {train_time:.1f}s | "
            f"Train Acc: {global_correct:.2f}% | Val Acc: {global_v_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e} ====\n"
        )
                        
        scheduler.step()
        
        # Save best model + periodic checkpoints
        if global_v_acc > best_acc:
            best_acc = global_v_acc
            if xm.is_master_ordinal():
                xm.save(model.state_dict(), "csmamba_v2_best.pt")
                xm.master_print(f"  💾 New best model saved! Val Acc: {best_acc:.2f}%")
                        
        if xm.is_master_ordinal() and (epoch % 10 == 0 or epoch == flags.epochs):
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': global_v_acc,
                'best_acc': best_acc,
            }
            xm.save(state, f"csmamba_v2_epoch{epoch}.pt")

    xm.master_print(f"\n🏁 Training Complete! Best Val Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    flags = parse_args()
    try:
        xmp.spawn(_mp_fn, args=(flags,), nprocs=8, start_method='fork')
    except KeyboardInterrupt:
        print("Training interrupted.")
