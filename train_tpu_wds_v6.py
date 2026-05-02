"""
CS-Mamba V6 — Characteristic Mamba TPU Training Script
======================================================
Selective 2D characteristic transport with learned divergence-free flow.
No torch.fft, no torch.complex, TPU/XLA-friendly.

Recommended usage:
  python train_tpu_wds_v6.py --dataset tiny-imagenet

If you want global XLA BF16 remapping, set it outside the script before launch:
  export XLA_USE_BF16=1
"""

import os
import sys
import threading
import time
import math
import argparse
import random
import gc
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("PJRT_DEVICE", "TPU")

_orig_excepthook = threading.excepthook


def _silent_dl_excepthook(args):
    if issubclass(args.exc_type, KeyError):
        return
    _orig_excepthook(args)


threading.excepthook = _silent_dl_excepthook
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from models.characteristic_mamba_v6 import CSMamba_V6


@dataclass
class EmptyConfig:
    pass


def parse_args():
    p = argparse.ArgumentParser("CS-Mamba V6 — Characteristic Mamba TPU Training")

    # Dataset
    p.add_argument("--dataset", choices=["imagenet1k", "tiny-imagenet"], default="tiny-imagenet")
    p.add_argument("--train_shards", type=str, default="")
    p.add_argument("--val_shards", type=str, default="")

    # Model
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--d_embed", type=int, default=192)
    p.add_argument("--n_mamba_layers", type=int, default=8)
    p.add_argument("--K_steps", type=int, default=4)
    p.add_argument("--n_classes", type=int, default=200)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--n_flow_groups", type=int, default=4, help="Number of flow groups for transport")

    # Optimizer
    p.add_argument("--batch_size", type=int, default=128, help="Per TPU core")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--base_lr", type=float, default=1e-3, help="Reference LR for GlobalBS=1024")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--amp_bf16", action="store_true", help="Use torch.autocast('xla', dtype=torch.bfloat16)")

    # Regularization
    p.add_argument("--mixup_alpha", type=float, default=0.8)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--mixup_prob", type=float, default=0.5)
    p.add_argument("--label_smooth", type=float, default=0.1)

    # Checkpoint
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_dir", type=str, default=".")
    p.add_argument("--save_every", type=int, default=10)

    # Infra
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def mixup_data(images, labels, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0))
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], float(lam)


def cutmix_data(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bsz, _, h, w = images.shape
    index = torch.randperm(bsz)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / max(h * w, 1)
    return mixed, labels, labels[index], float(lam)


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def mixed_top1(logits, labels_a, labels_b, lam):
    pred = logits.argmax(dim=1)
    correct_a = pred.eq(labels_a).float()
    correct_b = pred.eq(labels_b).float()
    return (lam * correct_a + (1.0 - lam) * correct_b).sum().item()


def build_lr_scheduler(optimizer, flags, scaled_lr):
    warmup = flags.warmup_epochs
    total = flags.epochs
    min_lr = flags.min_lr

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / scaled_lr, cosine_factor)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_wds_loader(shards_url, batch_size, flags, is_training=True):
    try:
        import torch_xla.runtime as xr
        global_rank = xr.global_ordinal()
        global_world_size = xr.world_size()
    except Exception:
        global_rank = 0
        global_world_size = 1

    def safe_xla_nodesplitter(urls):
        urls = list(urls)
        return urls[global_rank::global_world_size]

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

    def apply_transforms(sample):
        return transform(sample[0]), sample[1]

    def apply_mix(sample):
        images, labels = sample
        labels = labels.long() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        # [ABLATION] Disabled mixup and cutmix
        # if np.random.random() < flags.mixup_prob:
        #     mixed, la, lb, lam = mixup_data(images, labels, flags.mixup_alpha)
        # else:
        #     mixed, la, lb, lam = cutmix_data(images, labels, flags.cutmix_alpha)
        # return mixed, la, lb, torch.tensor(lam, dtype=torch.float32)
        return images, labels, labels, torch.tensor(1.0, dtype=torch.float32)

    def apply_stack_val(sample):
        images, labels = sample
        labels = labels.long() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        return images, labels

    dataset = (
        wds.WebDataset(
            shards_url, 
            resampled=is_training, 
            nodesplitter=safe_xla_nodesplitter, 
            shardshuffle=False,
            empty_check=False,
        )
        .shuffle(5000 if is_training else 0)
        .decode("pil")
        .to_tuple("jpg;png", "cls")
        .map(apply_transforms)
        .batched(batch_size, partial=False)
    )
    dataset = dataset.map(apply_mix if is_training else apply_stack_val)
    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=flags.num_workers,
        pin_memory=True,
        prefetch_factor=2 if flags.num_workers > 0 else None,
        persistent_workers=False,
    )


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image, label = item["image"], item["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
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
        # [ABLATION] Disabled mixup and cutmix
        # if np.random.random() < flags.mixup_prob:
        #     mixed, la, lb, lam = mixup_data(images, labels, flags.mixup_alpha)
        # else:
        #     mixed, la, lb, lam = cutmix_data(images, labels, flags.cutmix_alpha)
        # return mixed, la, lb, torch.tensor(lam, dtype=torch.float32)
        return images, labels, labels, torch.tensor(1.0, dtype=torch.float32)

    dataset = TinyImageNetDataset(ds["train"] if is_training else ds["valid"], transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=xr.world_size(),
        rank=xm.get_ordinal(),
        shuffle=is_training,
    )
    return DataLoader(
        dataset,
        batch_size=flags.batch_size,
        sampler=sampler,
        num_workers=flags.num_workers,
        drop_last=is_training,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=mixup_collate if is_training else None,
    )


def _autocast_ctx(flags):
    if flags.amp_bf16:
        return torch.autocast(device_type="xla", dtype=torch.bfloat16)
    return torch.autocast(device_type="xla", enabled=False)


def _mp_fn(index, flags):
    device = xm.xla_device()
    torch.manual_seed(flags.seed + index)
    np.random.seed(flags.seed + index)
    random.seed(flags.seed + index)

    if xm.is_master_ordinal():
        os.makedirs(flags.save_dir, exist_ok=True)

    cfg = EmptyConfig()
    cfg.img_size = flags.img_size
    cfg.patch_size = flags.patch_size
    cfg.d_embed = flags.d_embed
    cfg.n_mamba_layers = flags.n_mamba_layers
    cfg.K_steps = flags.K_steps
    cfg.n_classes = flags.n_classes
    cfg.canvas_size = flags.img_size
    cfg.n_flow_groups = flags.n_flow_groups

    model = CSMamba_V6(cfg).to(device)

    world_size = xr.world_size()
    global_bs = flags.batch_size * world_size
    scaled_lr = flags.base_lr * (global_bs / 1024.0)

    criterion = nn.CrossEntropyLoss(label_smoothing=flags.label_smooth)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=flags.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = build_lr_scheduler(optimizer, flags, scaled_lr)

    start_epoch = 1
    best_acc = 0.0
    if flags.resume:
        xm.master_print(f"Resuming from: {flags.resume}")
        ckpt = torch.load(flags.resume, map_location="cpu")
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_acc = ckpt.get("best_acc", 0.0)
        else:
            model.load_state_dict(ckpt)

    is_imagenet = flags.dataset == "imagenet1k"
    if is_imagenet:
        train_loader = build_wds_loader(flags.train_shards, flags.batch_size, flags, True)
        val_loader = build_wds_loader(flags.val_shards, flags.batch_size, flags, False)
        train_steps = math.ceil(1_281_167 / global_bs)
        val_steps = math.ceil(50_000 / global_bs)
    else:
        train_loader = build_tiny_loader(flags, True)
        val_loader = build_tiny_loader(flags, False)
        train_steps = math.ceil(100_000 / global_bs)
        val_steps = math.ceil(10_000 / global_bs)

    n_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"\n{'='*72}")
    xm.master_print(f"CS-Mamba V6 (Characteristic Mamba) | Params: {n_params/1e6:.1f}M")
    xm.master_print(f"World Size: {world_size} TPU cores | Global BS: {global_bs}")
    xm.master_print(f"Scaled LR: {scaled_lr:.6f} | AMP BF16: {flags.amp_bf16} | Flow Groups: {flags.n_flow_groups}")
    xm.master_print(f"{'='*72}\n")

    for epoch in range(start_epoch, flags.epochs + 1):
        if not is_imagenet:
            train_loader.sampler.set_epoch(epoch)

        pl_train = pl.ParallelLoader(train_loader, [device])
        para_train = pl_train.per_device_loader(device)

        model.train()
        tracker = xm.RateTracker()
        total_loss, total_correct, total = 0.0, 0.0, 0
        t0 = time.time()
        
        xm.master_print(f"=== Starting Epoch {epoch}. First step will take 5-10 minutes for XLA Compilation! ===")

        for step, batch in enumerate(para_train):
            if step >= train_steps:
                break
            images, labels_a, labels_b, lam = batch
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(flags):
                logits = model(images)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss.backward()
            xm.reduce_gradients(optimizer)
            # [ABLATION] Commented out gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), flags.grad_clip)
            xm.optimizer_step(optimizer)
            tracker.add(flags.batch_size)

            total_loss += loss.item() * images.size(0)
            total_correct += mixed_top1(logits, labels_a, labels_b, float(lam))
            total += images.size(0)

            if step > 0 and step % (50 if is_imagenet else 10) == 0:
                xm.master_print(
                    f"E{epoch:03d} | Step {step}/{train_steps} | "
                    f"Loss {loss.item():.4f} | Rate {tracker.global_rate():.1f} img/s"
                )

        train_time = time.time() - t0
        g_total = max(xm.mesh_reduce("tr_n", total, np.sum), 1)
        g_loss = xm.mesh_reduce("tr_loss", total_loss, np.sum) / g_total
        g_acc = 100.0 * xm.mesh_reduce("tr_c", total_correct, np.sum) / g_total

        del para_train, pl_train
        gc.collect()

        pl_val = pl.ParallelLoader(val_loader, [device])
        para_val = pl_val.per_device_loader(device)

        model.eval()
        v_correct, v_total = 0, 0
        t1 = time.time()
        for step, batch in enumerate(para_val):
            if step >= val_steps:
                break
            images, labels = batch
            with torch.no_grad():
                with _autocast_ctx(flags):
                    logits = model(images)
            v_correct += logits.argmax(1).eq(labels).sum().item()
            v_total += images.size(0)
            xm.mark_step()

        val_time = time.time() - t1
        v_acc = 100.0 * xm.mesh_reduce("v_c", v_correct, np.sum) / max(xm.mesh_reduce("v_n", v_total, np.sum), 1)
        scheduler.step()

        del para_val, pl_val
        gc.collect()

        xm.master_print(
            f"\nEpoch {epoch:03d}/{flags.epochs} | "
            f"Train {g_acc:.1f}% (loss {g_loss:.4f}) | Val {v_acc:.1f}% | "
            f"Train {train_time:.0f}s | Val {val_time:.0f}s | LR {scheduler.get_last_lr()[0]:.2e}\n"
        )

        # Checkpointing
        if v_acc > best_acc:
            best_acc = v_acc

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "val_acc": v_acc,
        }
        latest_path = os.path.join(flags.save_dir, f"csmamba_v6_{flags.dataset}_latest.pt")

        xm.save(ckpt, latest_path, master_only=True, global_master=True)

        if abs(v_acc - best_acc) < 1e-12:
            best_path = os.path.join(flags.save_dir, f"csmamba_v6_{flags.dataset}_best.pt")
            xm.save(ckpt, best_path, master_only=True, global_master=True)
            xm.master_print(f"New best! Val Acc: {best_acc:.2f}% saved to {best_path}")

    xm.master_print(f"Training complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    flags = parse_args()
    if flags.dataset == "tiny-imagenet":
        from datasets import load_dataset
        print("[Main] Pre-fetching Tiny-ImageNet cache...")
        load_dataset("Maysee/tiny-imagenet")
    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method="fork")
