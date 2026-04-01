"""
TPU V4-8 WebDataset Streaming Script for CG-Mamba [BASE SCALED MODEL]
======================================================================
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

from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba

def parse_args():
    p = argparse.ArgumentParser()
    
    # ── Dataset URLs ──
    p.add_argument('--train_shards', type=str, required=True,
                   help='e.g., /home/user/datasets/imagenet_hf/imagenet-1k-train-{0000..1023}.tar')
    p.add_argument('--val_shards', type=str, required=True,
                   help='e.g., /home/user/datasets/imagenet_hf/imagenet-1k-valid-{0000..0041}.tar')
                   
    # ── Scaled Model Geometry (ViT-Small/Base Equivalency) ──
    p.add_argument('--img_size', type=int, default=224, help="ImageNet standard resolution")
    p.add_argument('--patch_size', type=int, default=16, help="Standard 16x16 patching")
    p.add_argument('--d_embed', type=int, default=384, help="Increased from 128 to 384 for SOTA capacity")
    p.add_argument('--d_state', type=int, default=16)
    p.add_argument('--n_mamba_layers', type=int, default=12, help="Increased from 4 to 12 blocks")
    p.add_argument('--K_steps', type=int, default=3, help="Keep at 3 for blazing fast throughput!")
    p.add_argument('--n_classes', type=int, default=1000)
    
    # ── Training Params ──
    p.add_argument('--batch_size', type=int, default=64, help="Batch size PER TPU CORE")
    p.add_argument('--epochs', type=int, default=300, help="Full standard ImageNet schedule")
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    
    return p.parse_args()


class EmptyConfig:
    pass


def build_wds_loader(shards_url, batch_size, is_training=True):
    if is_training:
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
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

    dataset = (
        wds.WebDataset(shards_url, resampled=True, nodesplitter=wds.split_by_worker)
        .shuffle(1000 if is_training else 0)
        .decode("pil")
        .to_tuple("jpg;png", "cls")
        .map(apply_transforms)
        .batched(batch_size, partial=False)
    )
    
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=4, pin_memory=False)
    return loader


def _mp_fn(index, flags):
    device = xm.xla_device()
    
    # ── Model Config mapping ─────────────────────────────────
    cfg = EmptyConfig()
    cfg.img_size = flags.img_size
    cfg.patch_size = flags.patch_size
    cfg.d_embed = flags.d_embed
    cfg.d_state = flags.d_state
    cfg.n_mamba_layers = flags.n_mamba_layers
    cfg.K_steps = flags.K_steps
    cfg.n_classes = flags.n_classes
    cfg.canvas_size = flags.img_size

    # ── Init Model ───────────────────────────────────────────
    model = CSMamba(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=flags.epochs, eta_min=1e-6)
    
    # ── Loaders ──────────────────────────────────────────────
    train_loader = build_wds_loader(flags.train_shards, flags.batch_size, is_training=True)
    val_loader   = build_wds_loader(flags.val_shards,   flags.batch_size, is_training=False)
    
    global_batch_size = flags.batch_size * xm.xrt_world_size() 
    train_steps = ceil(1281167 / global_batch_size)
    val_steps   = ceil(50000   / global_batch_size)
    
    para_train = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    para_val   = pl.ParallelLoader(val_loader,   [device]).per_device_loader(device)
    
    xm.master_print(f"\n🚀 Launching CSMamba Base (Params: {sum(p.numel() for p in model.parameters()):,})")
    xm.master_print(f"Global Batch Size: {global_batch_size} (64/core on {xm.xla_real_devices([device])})")
    xm.master_print(f"Train Steps/Epoch: {train_steps} | Val Steps: {val_steps}")
    
    for epoch in range(1, flags.epochs + 1):
        
        # ── 1. TRAINING PHASE ────────────────────────────────────
        model.train()
        tracker = xm.RateTracker()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        
        for step, (images, labels) in enumerate(para_train):
            if step >= train_steps: break
                
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(flags.batch_size)
            
            total_loss += loss.item() * images.size(0)
            correct += logits.argmax(dim=1).eq(labels).sum().item()
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
        
        for step, (images, labels) in enumerate(para_val):
            if step >= val_steps: break
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
                        
        if xm.is_master_ordinal():
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': global_v_acc
            }
            xm.save(state, f"cgmamba_base_imagenet_epoch{epoch}.pt")

if __name__ == '__main__':
    flags = parse_args()
    try:
        xmp.spawn(_mp_fn, args=(flags,), nprocs=8, start_method='fork')
    except KeyboardInterrupt:
        print("Training interrupted.")
