"""
TPU V4-8 WebDataset Streaming Script for CG-Mamba
==================================================
This script solves the 100GB local disk limit of TPU VMs by streaming ImageNet 
shards directly from a mounted Google Cloud Storage (gcsfuse) bucket.

Multi-core parallelization is handled by torch_xla.distributed.xla_multiprocessing,
where all 8 cores independently stream their own batch of .tar shards.

Usage on the TPU VM:
    pip install webdataset torch_xla
    python3 train_tpu_wds.py --shards_url "~/datasets/imagenet_hf/imagenet-1k-train-{0000..1023}.tar"
"""

import os
import time
import argparse
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds

# TPU XLA Imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from models.continuous_graph_mamba import ContinuousGraphMambaClassifier as CGMamba

def parse_args():
    p = argparse.ArgumentParser()
    # WebDataset Shards URL (Point to your gcsfuse mount!)
    p.add_argument('--shards_url', type=str, required=True,
                   help='e.g., /home/user/datasets/imagenet_hf/imagenet-1k-train-{0000..1023}.tar')
                   
    # Model geometry (Matched to generic CG-Mamba settings from train.py)
    p.add_argument('--img_size', type=int, default=224, help="ImageNet standard resolution")
    p.add_argument('--patch_size', type=int, default=16, help="Standard 16x16 patching")
    p.add_argument('--d_embed', type=int, default=128)
    p.add_argument('--d_state', type=int, default=16)
    p.add_argument('--n_mamba_layers', type=int, default=4)
    p.add_argument('--K_steps', type=int, default=3)
    p.add_argument('--n_classes', type=int, default=1000)
    
    # Training Params
    p.add_argument('--batch_size', type=int, default=128, help="Batch size PER TPU CORE")
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    
    return p.parse_args()


class EmptyConfig:
    pass


def build_streaming_dataloader(shards_url, batch_size, is_training=True):
    """
    Builds the WebDataset pipeline.
    `resampled=True` allows multiple TPU cores to draw endless unique samples from the shards
    without complex split-by-node logic.
    """
    if is_training:
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    )
    
    # We must explicitly batch inside WebDataset for efficient pipeline streaming
    dataset = dataset.batched(batch_size, partial=False)
    
    loader = wds.WebLoader(
        dataset, 
        batch_size=None, # Already batched by dataset.batched
        num_workers=4,   # 4 workers fetching shards from Cloud Storage locally
        pin_memory=False # Keep False for PyTorch XLA!
    )
    return loader


def _mp_fn(index, flags):
    """
    This function is spawned 8 times by xmp.spawn() - once for each TPU Core.
    `index` represents the local core ID (0-7).
    """
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
    cfg.canvas_size = flags.img_size  # spatial compat

    # ── Init Model & move to specific TPU core ───────────────────
    model = CGMamba(cfg).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    
    # ── WebDataset Loader ────────────────────────────────────────
    loader = build_streaming_dataloader(flags.shards_url, flags.batch_size)
    
    # XLA Parallel Loader wrapper to keep TPU fed efficiently
    # Note: Since WebDataset resampled=True is infinite, we define 'steps_per_epoch' explicitly.
    # Total ImageNet Train size approx 1,281,167. 
    # With 8 cores (batch_sz per core = N), effective global batch = 8*N
    global_batch_size = flags.batch_size * xm.xrt_world_size() 
    steps_per_epoch = ceil(1281167 / global_batch_size)
    
    para_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    
    xm.master_print(f"\n🚀 Starting TPU Training on Model (Params: {sum(p.numel() for p in model.parameters()):,})")
    xm.master_print(f"Device topology: {xm.xla_real_devices(device)}")
    xm.master_print(f"Effective Global Batch Size: {global_batch_size}")
    xm.master_print(f"Steps per epoch: {steps_per_epoch}")
    
    for epoch in range(1, flags.epochs + 1):
        model.train()
        tracker = xm.RateTracker()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        t0 = time.time()
        for step, (images, labels) in enumerate(para_loader):
            if step >= steps_per_epoch:
                break # Manually end "epochs" since stream is infinite
                
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            
            # The Magic TPU synchronization step
            xm.optimizer_step(optimizer)
            tracker.add(flags.batch_size)
            
            # Track metrics locally
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
            
            # Print periodic tracking from Core 0
            if step % 50 == 0:
                xm.master_print(
                    f"Epoch {epoch} | Step {step}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Rate: {tracker.rate():.2f} samples/sec | "
                    f"Global Rate: {tracker.global_rate():.2f} samples/sec"
                )

        # Synchronize and log end-of-epoch accuracy across all 8 cores
        # xm.mesh_reduce calculates the true global sum
        global_loss = xm.mesh_reduce("loss_reduce", total_loss, np.sum)
        global_correct = xm.mesh_reduce("correct_reduce", correct, np.sum)
        global_total = xm.mesh_reduce("total_reduce", total, np.sum)
        
        epoch_loss = global_loss / global_total
        epoch_acc = 100.0 * global_correct / global_total
        elapsed = time.time() - t0
        
        xm.master_print(f"==== Epoch {epoch} Completed in {elapsed:.1f}s | "
                        f"Avg Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% ====")
                        
        # Save explicitly from Core 0
        if xm.is_master_ordinal():
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            xm.save(state, f"cgmamba_imagenet_epoch{epoch}.pt")


if __name__ == '__main__':
    # 8 Cores natively available in TPU v4-8
    flags = parse_args()
    try:
        xmp.spawn(_mp_fn, args=(flags,), nprocs=8, start_method='fork')
    except KeyboardInterrupt:
        print("Training interrupted.")
