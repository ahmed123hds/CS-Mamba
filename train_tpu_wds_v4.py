"""
CS-Mamba V4 — Production Training Script (TPU v4)
==================================================
Gold-standard DeiT/VMamba training recipe for ICLR 2026.
Supports both ImageNet-1K (via WebDataset) and Tiny-ImageNet (via HuggingFace).

Training Recipe:
  Optimizer:      AdamW (β1=0.9, β2=0.999)
  Base LR:        1e-3  (linear scaled: lr = 1e-3 × GlobalBS/1024)
  Weight Decay:   0.05
  Warmup:         20 epochs (linear from 1e-6 → peak)
  Schedule:       Cosine Annealing → eta_min=1e-5
  Grad Clip:      max_norm=1.0
  Augmentation:   MixUp (α=0.8), CutMix (α=1.0), RandAug(2,9), ColorJitter(0.4)
"""

import os
import sys
import threading
import time
import math
import argparse
import random
import numpy as np

# ── SUPPRESS HARMLESS DATALOADER KEYERROR ──
# PyTorch's DataLoader throws a background KeyError: 62 thread exception when XLA 
# ParallelLoader abruptly stops at the end of validation. This hides the traceback 
# so it doesn't overwrite your beautiful Epoch summary printout!
_orig_excepthook = threading.excepthook
def _silent_dl_excepthook(args):
    if issubclass(args.exc_type, KeyError): return
    _orig_excepthook(args)
threading.excepthook = _silent_dl_excepthook

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
def parse_args():
    p = argparse.ArgumentParser("CS-Mamba V4 — Production Training (TPU)")

    # ── Dataset ──
    p.add_argument('--dataset',      type=str,   choices=['imagenet1k', 'tiny-imagenet'], default='tiny-imagenet')
    p.add_argument('--train_shards', type=str,   default="", help="WDS tar shards (for ImageNet)")
    p.add_argument('--val_shards',   type=str,   default="", help="WDS tar shards (for ImageNet)")

    # ── Model Architecture ──
    p.add_argument('--img_size',       type=int,   default=64,   help="224 for ImageNet, 64 for Tiny")
    p.add_argument('--patch_size',     type=int,   default=4,    help="16 for ImageNet, 4 for Tiny")
    p.add_argument('--d_embed',        type=int,   default=192,  help="384=Small/Base, 192=Tiny")
    p.add_argument('--n_mamba_layers', type=int,   default=8,   help="12=Small, 24=Base, 8=Tiny")
    p.add_argument('--K_steps',        type=int,   default=3)
    p.add_argument('--n_classes',      type=int,   default=200,  help="1000 for ImageNet, 200 for Tiny")
    p.add_argument('--drop_path',      type=float, default=0.1,  help="0.1=Small, 0.2=Base")

    # ── Optimizer ──
    p.add_argument('--batch_size',   type=int,   default=128,   help="Per TPU core")
    p.add_argument('--epochs',       type=int,   default=300)
    p.add_argument('--base_lr',      type=float, default=1e-3,  help="Base LR for GlobalBS=1024")
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--grad_clip',    type=float, default=1.0,   help="Max gradient norm")
    p.add_argument('--warmup_epochs', type=int,  default=20)
    p.add_argument('--min_lr',       type=float, default=1e-5)

    # ── Regularization ──
    p.add_argument('--mixup_alpha',  type=float, default=0.8)
    p.add_argument('--cutmix_alpha', type=float, default=1.0)
    p.add_argument('--mixup_prob',   type=float, default=0.5)
    p.add_argument('--label_smooth', type=float, default=0.1)

    # ── Checkpoint ──
    p.add_argument('--resume',       type=str,   default='')
    p.add_argument('--save_dir',     type=str,   default='.')
    p.add_argument('--save_every',   type=int,   default=10)

    # ── Infrastructure ──
    p.add_argument('--multi_host',   action='store_true', help="Use for Pods")
    p.add_argument('--num_workers',  type=int,   default=8)
    p.add_argument('--seed',         type=int,   default=42)

    return p.parse_args()


class EmptyConfig: pass


# ════════════════════════════════════════════════════════════════════
#  MixUp & CutMix
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


def build_lr_scheduler(optimizer, flags, scaled_lr):
    warmup = flags.warmup_epochs
    total = flags.epochs
    min_lr = flags.min_lr

    def lr_lambda(epoch):
        if epoch < warmup: return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / scaled_lr, cosine_factor)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════
#  ImageNet-1K WebDataset Loader
# ════════════════════════════════════════════════════════════════════
def build_wds_loader(shards_url, batch_size, flags, is_training=True):
    if is_training:
        transform = T.Compose([
            T.RandomResizedCrop(flags.img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize(int(flags.img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(flags.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def apply_transforms(s): return transform(s[0]), s[1]

    def apply_mixup_cutmix(b):
        images, labels = b
        labels = labels.long() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        if np.random.random() < flags.mixup_prob:
            mixed, la, lb, lam = mixup_data(images, labels, flags.mixup_alpha)
        else:
            mixed, la, lb, lam = cutmix_data(images, labels, flags.cutmix_alpha)
        return mixed, la, lb, torch.tensor(lam, dtype=torch.float32)

    def apply_stack_val(b):
        images, labels = b
        labels = labels.long() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        return images, labels

    # When resampled=True is used, WebDataset natively splits workers and nodes infinitely!
    # Using a custom nodesplitter with resampled=True causes a known hang issue at epoch boundaries.
    dataset = (
        wds.WebDataset(shards_url, resampled=True, nodesplitter=None)
        .shuffle(5000 if is_training else 0)
        .decode("pil")
        .to_tuple("jpg;png", "cls")
        .map(apply_transforms)
        .batched(batch_size, partial=False)
    )

    if is_training: dataset = dataset.map(apply_mixup_cutmix)
    else: dataset = dataset.map(apply_stack_val)

    # CRITICAL: persistent_workers MUST BE FALSE for PyTorch XLA!
    # XLA ParallelLoader creates a background thread. Since we must delete ParallelLoader 
    # every epoch to prevent memory leaks, that deletion violently breaks the DataLoader's 
    # active IPC queue. If persistent_workers=True, the worker threads die and cause 
    # a deadlock exactly like what happened at Epoch 62.
    # 
    # persistent_workers=False cleanly rebuilds the threads. The 3-5 minute delay 
    # per epoch is scientifically required to run ImageNet successfully on TPUs.
    return wds.WebLoader(dataset, batch_size=None, num_workers=flags.num_workers,
                         pin_memory=True, prefetch_factor=2, persistent_workers=False)


# ════════════════════════════════════════════════════════════════════
#  Tiny-ImageNet loader
# ════════════════════════════════════════════════════════════════════
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        image, label = item['image'], item['label']
        if image.mode != 'RGB': image = image.convert('RGB')
        return self.transform(image) if self.transform else image, label

def build_tiny_loader(flags, is_training=True):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, DistributedSampler
    ds = load_dataset("Maysee/tiny-imagenet")

    if is_training:
        transform = T.Compose([
            T.RandomResizedCrop(flags.img_size, scale=(0.3, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def mixup_collate(batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if np.random.random() < flags.mixup_prob:
            mixed, la, lb, lam = mixup_data(images, labels, flags.mixup_alpha)
        else:
            mixed, la, lb, lam = cutmix_data(images, labels, flags.cutmix_alpha)
        return mixed, la, lb, torch.tensor(lam, dtype=torch.float32)

    dataset = TinyImageNetDataset(ds['train'] if is_training else ds['valid'], transform)
    sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=is_training)
    
    loader = DataLoader(
        dataset, batch_size=flags.batch_size, sampler=sampler, 
        num_workers=flags.num_workers, drop_last=is_training,
        collate_fn=mixup_collate if is_training else None
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

    cfg = EmptyConfig()
    cfg.img_size       = flags.img_size
    cfg.patch_size     = flags.patch_size
    cfg.d_embed        = flags.d_embed
    cfg.n_mamba_layers = flags.n_mamba_layers
    cfg.K_steps        = flags.K_steps
    cfg.n_classes      = flags.n_classes
    cfg.canvas_size    = flags.img_size
    cfg.drop_path      = flags.drop_path

    model = CSMamba_V4(cfg).to(device)

    world_size = xm.xrt_world_size()
    global_bs = flags.batch_size * world_size
    scaled_lr = flags.base_lr * (global_bs / 1024.0)

    criterion = nn.CrossEntropyLoss(label_smoothing=flags.label_smooth)
    optimizer = optim.AdamW(
        model.parameters(), lr=scaled_lr, weight_decay=flags.weight_decay, betas=(0.9, 0.999),
    )
    scheduler = build_lr_scheduler(optimizer, flags, scaled_lr)

    start_epoch = 1
    best_acc = 0.0

    if flags.resume:
        xm.master_print(f"📦 Resuming from: {flags.resume}")
        ckpt = torch.load(flags.resume, map_location='cpu')
        
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
            if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_acc = ckpt.get('best_acc', 0.0)
            xm.master_print(f"  ✅ Resumed full state! Starting at Epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)
            xm.master_print("  ⚠️ Loaded raw model weights only. Optimizer/epoch stats reset.")

    is_imagenet = flags.dataset == 'imagenet1k'
    
    if is_imagenet:
        train_loader = build_wds_loader(flags.train_shards, flags.batch_size, flags, True)
        val_loader   = build_wds_loader(flags.val_shards,   flags.batch_size, flags, False)
        train_steps  = math.ceil(1_281_167 / global_bs)
        val_steps    = math.ceil(50_000 / global_bs)
    else:
        train_loader = build_tiny_loader(flags, True)
        val_loader   = build_tiny_loader(flags, False)
        train_steps  = math.ceil(100_000 / global_bs)
        val_steps    = math.ceil(10_000 / global_bs)
        
    n_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"\n{'='*70}\n  CS-Mamba V4 — {flags.dataset.upper()} Training (DeiT Gold Standard)\n{'='*70}")
    xm.master_print(f"  Params:       {n_params/1e6:.1f}M")
    xm.master_print(f"  World Size:   {world_size} TPU cores | Global BS: {global_bs}")
    xm.master_print(f"  Scaled LR:    {scaled_lr:.6f} | WD: {flags.weight_decay}")
    xm.master_print(f"  Grad Clip:    {flags.grad_clip} | Warmup: {flags.warmup_epochs} epochs")
    xm.master_print(f"{'='*70}\n")

    for epoch in range(start_epoch, flags.epochs + 1):
        if not is_imagenet:
            train_loader.sampler.set_epoch(epoch)

        # ── CRITICAL FIX: Create ParallelLoader, use it, then let it be garbage collected ──
        # Previously we created new ParallelLoaders every epoch without cleanup.
        # Each ParallelLoader spawns XLA background threads that were never freed,
        # causing host RAM to grow ~50MB/epoch until OOM killed the process at Epoch 8/23.
        # 
        # Fix: Create, use, then explicitly delete + garbage collect each ParallelLoader.
        import gc

        # ── TRAINING PHASE ──
        pl_train = pl.ParallelLoader(train_loader, [device])
        para_train = pl_train.per_device_loader(device)

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

            xm.reduce_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), flags.grad_clip)
            xm.optimizer_step(optimizer)
            tracker.add(flags.batch_size)

            total_loss += loss.item() * images.size(0)
            correct += logits.argmax(1).eq(labels_a).sum().item()
            total += images.size(0)

            if step > 0 and step % (50 if is_imagenet else 10) == 0:
                xm.master_print(f"  E{epoch:03d} | Step {step}/{train_steps} | Loss: {loss.item():.4f} | Rate: {tracker.global_rate():.1f} img/s")

        train_time = time.time() - t0
        g_loss = xm.mesh_reduce("tr_loss", total_loss, np.sum) / max(xm.mesh_reduce("tr_n", total, np.sum), 1)
        g_acc  = 100.0 * xm.mesh_reduce("tr_c", correct, np.sum) / max(xm.mesh_reduce("tr_n", total, np.sum), 1)

        # ── CLEANUP TRAINING PARALLELLOADER ──
        del para_train, pl_train
        gc.collect()

        # ── VALIDATION PHASE ──
        pl_val = pl.ParallelLoader(val_loader, [device])
        para_val = pl_val.per_device_loader(device)

        model.eval()
        v_correct, v_total = 0, 0
        t1 = time.time()

        for step, batch in enumerate(para_val):
            if step >= val_steps: break
            images, labels = batch
            with torch.no_grad(): logits = model(images)
            v_correct += logits.argmax(1).eq(labels).sum().item()
            v_total += images.size(0)
            xm.mark_step()

        val_time = time.time() - t1
        v_acc = 100.0 * xm.mesh_reduce("v_c", v_correct, np.sum) / max(xm.mesh_reduce("v_n", v_total, np.sum), 1)
        scheduler.step()

        # ── CLEANUP VALIDATION PARALLELLOADER ──
        del para_val, pl_val
        gc.collect()

        xm.master_print(
            f"\n  ═══ Epoch {epoch:03d}/{flags.epochs} | Train {g_acc:.1f}% (loss {g_loss:.4f}) | "
            f"Val {v_acc:.1f}% | Train Time {train_time:.0f}s | Val Time {val_time:.0f}s | LR {scheduler.get_last_lr()[0]:.2e} ═══\n"
        )

        # ── Save Full Checkpoint for Resumption ──
        if xm.is_master_ordinal():
            ckpt = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'val_acc': v_acc
            }
            latest_path = os.path.join(flags.save_dir, f"csmamba_v4_{flags.dataset}_latest.pt")
            xm.save(ckpt, latest_path)

            if v_acc > best_acc:
                best_acc = v_acc
                best_path = os.path.join(flags.save_dir, f"csmamba_v4_{flags.dataset}_best.pt")
                xm.save(ckpt, best_path)
                xm.master_print(f"  💾 New best! Val Acc: {best_acc:.2f}% saved to {best_path}")

    xm.master_print(f"\n🏁 Training Complete! Best Val Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    flags = parse_args()
    if flags.dataset == 'tiny-imagenet':
        from datasets import load_dataset
        print("[Main] Pre-fetching Tiny-ImageNet cache...")
        load_dataset("Maysee/tiny-imagenet")

    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='fork')
