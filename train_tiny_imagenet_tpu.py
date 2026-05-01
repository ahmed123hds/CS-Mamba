"""
Tiny-ImageNet CS-Mamba Ablation — TPU v4-8 Optimized
=====================================================
Adapted from train_tiny_imagenet_v2.py for 8-core TPU training.
Uses HuggingFace datasets (auto-downloaded), XLA ParallelLoader,
and DeiT-level regularization.

Usage (on TPU VM):
  python train_tiny_imagenet_tpu.py --mode v6         # V6 only
  python train_tiny_imagenet_tpu.py --mode vmamba     # VMamba-style 4D scan baseline
  python train_tiny_imagenet_tpu.py --mode v6_vmamba  # V6 vs VMamba baseline
  python train_tiny_imagenet_tpu.py --mode v2         # V2 only
  python train_tiny_imagenet_tpu.py --mode v1         # V1 only
  python train_tiny_imagenet_tpu.py --mode compare    # Head-to-head V1-V6
"""

import os, sys, time, math, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Keep the TPU path on PyTorch/XLA only. Dynamo/Inductor can accidentally
# spawn CPU compile workers and stall before the first XLA step.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.datasets as datasets

from datasets import load_dataset

# TPU XLA Imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# Model Imports
from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba_V1
from models.continuous_spatial_mamba_v2 import CSMamba_V2
from models.continuous_spatial_mamba_v3 import CSMamba_V3
from models.continuous_spatial_mamba_v4 import CSMamba_V4
from models.characteristic_mamba_v6 import CSMamba_V6
from models.vmamba_4d import VMamba4D


# ─────────────────────────────────────────────────────────
#  CPU-side Mixup + CutMix Collate (XLA-safe)
# ─────────────────────────────────────────────────────────
class MixupCutMixCollate:
    """
    Performs Mixup/CutMix entirely on CPU inside the DataLoader collate.

    Why this is the right pattern for XLA/TPU:
      - All random ops (lam, randperm, coords, mask) happen on CPU.
      - The TPU receives fixed-shape tensors every step:
          imgs:    (B, C, H, W)  float32
          targets: (B, n_classes) float32  ← soft labels
      - The XLA computation graph NEVER changes → compiles once, full speed.
      - Eliminates all host-device syncs and dynamic graph branches from the
        training step that were causing constant recompilation.

    CrossEntropyLoss accepts (B, C) float targets natively (PyTorch >= 1.10).
    """
    def __init__(self, n_classes: int, mixup_alpha: float = 0.8,
                 cutmix_alpha: float = 1.0, p_cutmix: float = 0.5):
        self.n_classes   = n_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.p_cutmix    = p_cutmix

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(labels), self.n_classes).scatter_(
            1, labels.unsqueeze(1), 1.0)

    def _mixup(self, imgs: torch.Tensor, labels: torch.Tensor):
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(len(imgs))  # CPU — avoids XLA s64 rng_uniform
        mixed = lam * imgs + (1.0 - lam) * imgs[idx]
        soft  = lam * self._one_hot(labels) + (1.0 - lam) * self._one_hot(labels[idx])
        return mixed, soft

    def _cutmix(self, imgs: torch.Tensor, labels: torch.Tensor):
        B, C, H, W = imgs.shape
        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        idx = torch.randperm(B)          # CPU — avoids XLA s64 rng_uniform
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat);  cut_h = int(H * cut_rat)
        cx = random.randint(0, W); cy = random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
        # Clone + slice entirely on CPU — TPU only receives the finished tensor
        mixed = imgs.clone()
        mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
        lam   = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
        soft  = lam * self._one_hot(labels) + (1.0 - lam) * self._one_hot(labels[idx])
        return mixed, soft

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs   = torch.stack(imgs)                             # (B, C, H, W)
        labels = torch.tensor(labels, dtype=torch.long)        # (B,)
        # [ABLATION] Commented out CutMix branch, only use Mixup
        # if random.random() < self.p_cutmix:
        #     return self._cutmix(imgs, labels)
        # else:
        return self._mixup(imgs, labels)


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
    running_total   = torch.tensor(0, dtype=torch.long, device=device)
    debug_first_step = getattr(cfg, "debug_first_step", False)
    grad_clip = getattr(cfg, "grad_clip", 1.0)

    for step, (imgs, targets) in enumerate(loader):
        if debug_first_step and step == 0:
            dbg_t = time.time()
            xm.master_print("    [debug] first batch received")

        optimizer.zero_grad()

        outs = model(imgs)
        # targets can be (B,) hard labels OR (B, C) soft labels from collate.
        # CrossEntropyLoss handles both natively (PyTorch >= 1.10).
        loss = criterion(outs, targets)

        loss.backward()

        # Gradient clipping for training stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        xm.optimizer_step(optimizer)
        if debug_first_step and step == 0:
            xm.master_print(f"    [debug] optimizer_step done ({time.time() - dbg_t:.2f}s)")

        tracker.add(cfg.batch_size)

        # Accuracy: for soft labels use argmax as proxy hard label
        hard = targets.argmax(1) if targets.dim() == 2 else targets
        running_correct += (outs.argmax(1) == hard).sum()
        running_total   += imgs.size(0)

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

    if getattr(cfg, 'cfar100', False) or getattr(cfg, 'cifar100', False):
        ds_name = "CIFAR-100 (100 classes, 50K images)"
    elif getattr(cfg, 'cfar10', False) or getattr(cfg, 'cifar10', False):
        ds_name = "CIFAR-10 (10 classes, 50K images)"
    else:
        ds_name = "Tiny-ImageNet (200 classes, 100K images)"

    xm.master_print(f"  Dataset:  {ds_name}")
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

    xm.master_print(f"\n  ✓ Best Val Acc: {best_acc:.2f}%\n")
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
    xm.master_print(f"  CS-Mamba V1-V6 — Tiny-ImageNet-200 Ablation (TPU)")
    xm.master_print(f"{'='*60}")
    xm.master_print(f"  V1: Collapsed diffusion (sum over S, manual slicing)")
    xm.master_print(f"  V2: State-preserving diffusion (per-S, F.conv2d)")
    xm.master_print(f"  VMamba4D: Fixed 4-route SS2D/Cross-Scan baseline")
    xm.master_print(f"  V6: Characteristic transport with self+8 routing")
    xm.master_print(f"{'='*60}\n")

    # ── Dataset: each spawned process reads from HF cache independently ──
    if flags.cfar100 or flags.cifar100:
        xm.master_print("  [Loading CIFAR-100...]")
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif flags.cfar10 or flags.cifar10:
        xm.master_print("  [Loading CIFAR-10...]")
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    else:
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
        transforms.Resize((flags.img_size, flags.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if flags.cfar100 or flags.cifar100:
        train_ds = datasets.CIFAR100(root='./torch_data', train=True, transform=transform_train)
        val_ds   = datasets.CIFAR100(root='./torch_data', train=False, transform=transform_val)
    elif flags.cfar10 or flags.cifar10:
        train_ds = datasets.CIFAR10(root='./torch_data', train=True, transform=transform_train)
        val_ds   = datasets.CIFAR10(root='./torch_data', train=False, transform=transform_val)
    else:
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

    # [ABLATION] Commented out Mixup/CutMix entirely
    # use_mixup = not getattr(flags, 'no_mixup', False)
    use_mixup = False
    mixup_collate = MixupCutMixCollate(n_classes=flags.n_classes) if use_mixup else None

    train_loader_raw = DataLoader(
        train_ds, batch_size=flags.batch_size,
        sampler=train_sampler, num_workers=0,
        drop_last=True, collate_fn=mixup_collate,
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
    setattr(flags, 'use_mixup', not getattr(flags, 'no_mixup', False))

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

    if flags.mode in ('v3', 'compare'):
        model_v3 = CSMamba_V3(flags).to(device)
        results['CSMamba_V3'] = train_model(
            'CSMamba_V3', model_v3, train_loader_raw, train_sampler, val_loader_raw, flags, device)
        del model_v3
        xm.mark_step()

    if flags.mode in ('v4', 'compare'):
        model_v4 = CSMamba_V4(flags).to(device)
        results['CSMamba_V4'] = train_model(
            'CSMamba_V4', model_v4, train_loader_raw, train_sampler, val_loader_raw, flags, device)
        del model_v4
        xm.mark_step()

    if flags.mode in ('vmamba', 'v6_vmamba', 'compare'):
        model_vmamba = VMamba4D(flags).to(device)
        results['VMamba4D'] = train_model(
            'VMamba4D', model_vmamba, train_loader_raw, train_sampler, val_loader_raw, flags, device)
        del model_vmamba
        xm.mark_step()

    if flags.mode in ('v6', 'v6_vmamba', 'compare'):
        model_v6 = CSMamba_V6(flags).to(device)
        results['CSMamba_V6'] = train_model(
            'CSMamba_V6', model_v6, train_loader_raw, train_sampler, val_loader_raw, flags, device)
        del model_v6
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
    p = argparse.ArgumentParser("CS-Mamba V1/V2/V3/V4/V6 + VMamba4D — Tiny-ImageNet (TPU)")
    p.add_argument(
        '--mode',
        choices=['v1', 'v2', 'v3', 'v4', 'v6', 'vmamba', 'v6_vmamba', 'compare'],
        default='v6',
    )
    p.add_argument('--img_size',       type=int,   default=64)
    p.add_argument('--patch_size',     type=int,   default=4)
    p.add_argument('--n_classes',      type=int,   default=200)
    p.add_argument('--d_embed',        type=int,   default=192)
    p.add_argument('--d_state',        type=int,   default=16)
    p.add_argument('--n_mamba_layers', type=int,   default=8)
    p.add_argument('--K_steps',        type=int,   default=3)
    p.add_argument('--n_flow_groups',  type=int,   default=4)
    p.add_argument('--drop_path',      type=float, default=0.1)
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch_size',     type=int,   default=128, help="Per TPU core")
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=0.05)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument(
        '--debug_first_step',
        action='store_true',
        help="Print first-step phase timings to diagnose TPU compile stalls.",
    )
    p.add_argument('--proj_drop',      type=float, default=0.1,
                   help="Projection dropout rate inside SSM and block (default: 0.1)")
    p.add_argument('--head_drop',      type=float, default=0.1,
                   help="Dropout rate before classification head (default: 0.1)")
    p.add_argument('--grad_clip',      type=float, default=1.0,
                   help="Max gradient norm for clipping (0 to disable, default: 1.0)")
    p.add_argument('--no_mixup',       action='store_true',
                   help="Disable Mixup/CutMix regularization")
                   
    # Dataset flags
    p.add_argument('--cifar100',       action='store_true', help="Use CIFAR-100 dataset")
    p.add_argument('--cfar100',        action='store_true', help="Alias for --cifar100")
    p.add_argument('--cifar10',        action='store_true', help="Use CIFAR-10 dataset")
    p.add_argument('--cfar10',         action='store_true', help="Alias for --cifar10")
    p.add_argument('--tinyimagenet',   action='store_true', help="Use Tiny-ImageNet (default)")
    
    return p.parse_args()


if __name__ == '__main__':
    flags = parse_args()

    # Pre-download dataset in main process so all spawned workers hit the local cache
    if flags.cfar100 or flags.cifar100:
        print("[Main] Pre-fetching CIFAR-100...")
        datasets.CIFAR100(root='./torch_data', train=True, download=True)
        datasets.CIFAR100(root='./torch_data', train=False, download=True)
        flags.n_classes = 100
    elif flags.cfar10 or flags.cifar10:
        print("[Main] Pre-fetching CIFAR-10...")
        datasets.CIFAR10(root='./torch_data', train=True, download=True)
        datasets.CIFAR10(root='./torch_data', train=False, download=True)
        flags.n_classes = 10
    else:
        print("[Main] Pre-fetching Tiny-ImageNet to local HuggingFace cache...")
        load_dataset("Maysee/tiny-imagenet")
        flags.n_classes = 200

    print("[Main] Dataset cached. Spawning TPU workers with start_method='fork'...")

    # Use 'fork' — inherits parent's /dev/accel0 TPU device permissions
    # Original fork issues (deadlock, slow training) were caused by:
    #   1. HuggingFace lock contention (fixed: pre-cached in main)
    #   2. CutMix dynamic shapes (fixed: removed from training loop)
    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='fork')
