#!/usr/bin/env bash
set -euo pipefail

# Debug one raw ImageNet sample against one CropDCT/FastQuant decoded sample.
#
# Required:
#   STORE_DIR=/path/to/cropdct_or_fast_quant_store
#
# Strongly recommended:
#   RAW_IMAGE=/path/to/original_image.jpg
#
# Optional:
#   TDCF_REPO=/path/to/tdcf_repo
#   STORE_CLASS=module.path:ClassName
#   IMAGE_ID=0
#   CROP_BOX=16,16,240,240
#   VIEW_SIZE=256
#   OUTPUT_SIZE=224
#   OUT_DIR=debug_cropdct_pipeline
#
# Example:
#   cd ~/model/CS-Mamba
#   TDCF_REPO=~/models/tdcf \
#   STORE_DIR=/mnt/dataset_disk/imagenet1k_fast_quant_dct_256/val \
#   RAW_IMAGE=/mnt/dataset_disk/imagenet_raw/val/n01440764/ILSVRC2012_val_00000293.JPEG \
#   IMAGE_ID=0 \
#   bash scripts/debug_cropdct_pipeline.sh

STORE_DIR="${STORE_DIR:-}"
RAW_IMAGE="${RAW_IMAGE:-}"
TDCF_REPO="${TDCF_REPO:-}"
STORE_CLASS="${STORE_CLASS:-}"
IMAGE_ID="${IMAGE_ID:-0}"
VIEW_SIZE="${VIEW_SIZE:-256}"
OUTPUT_SIZE="${OUTPUT_SIZE:-224}"
CROP_BOX="${CROP_BOX:-}"
OUT_DIR="${OUT_DIR:-debug_cropdct_pipeline}"

if [[ -z "${STORE_DIR}" ]]; then
  echo "ERROR: STORE_DIR is required." >&2
  echo "Example: STORE_DIR=/mnt/dataset_disk/imagenet1k_fast_quant_dct_256/val bash $0" >&2
  exit 2
fi

python - <<'PY'
import importlib
import inspect
import math
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image


STORE_DIR = Path(os.environ["STORE_DIR"]).expanduser()
RAW_IMAGE = os.environ.get("RAW_IMAGE", "").strip()
TDCF_REPO = os.environ.get("TDCF_REPO", "").strip()
STORE_CLASS = os.environ.get("STORE_CLASS", "").strip()
IMAGE_ID = int(os.environ.get("IMAGE_ID", "0"))
VIEW_SIZE = int(os.environ.get("VIEW_SIZE", "256"))
OUTPUT_SIZE = int(os.environ.get("OUTPUT_SIZE", "224"))
CROP_BOX_ENV = os.environ.get("CROP_BOX", "").strip()
OUT_DIR = Path(os.environ.get("OUT_DIR", "debug_cropdct_pipeline")).expanduser()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def fail(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def add_repo_to_path():
    if TDCF_REPO:
        repo = Path(TDCF_REPO).expanduser().resolve()
        if not repo.exists():
            fail(f"TDCF_REPO does not exist: {repo}")
        sys.path.insert(0, str(repo))
        print(f"[info] Added TDCF_REPO to sys.path: {repo}")
    else:
        print("[info] TDCF_REPO not set; importing store from current Python path.")


def import_store_class():
    if STORE_CLASS:
        if ":" not in STORE_CLASS:
            fail("STORE_CLASS must look like module.path:ClassName")
        mod_name, cls_name = STORE_CLASS.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)

    candidates = [
        ("cropdct_store", "CropDCTStore"),
        ("tdcf.cropdct_store", "CropDCTStore"),
        ("tdcf.data.cropdct_store", "CropDCTStore"),
        ("tdcf.stores.cropdct_store", "CropDCTStore"),
        ("fast_quant_store", "FastQuantStore"),
        ("tdcf.fast_quant_store", "FastQuantStore"),
        ("tdcf.data.fast_quant_store", "FastQuantStore"),
        ("tdcf.stores.fast_quant_store", "FastQuantStore"),
    ]
    errors = []
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, cls_name):
                print(f"[info] Using store class {mod_name}:{cls_name}")
                return getattr(mod, cls_name)
        except Exception as exc:
            errors.append(f"{mod_name}:{cls_name} -> {type(exc).__name__}: {exc}")

    print("[debug] Store import attempts failed:")
    for err in errors:
        print(f"  - {err}")
    fail(
        "Could not import a store class. Set STORE_CLASS=module.path:ClassName "
        "and TDCF_REPO=/path/to/tdcf."
    )


def make_store(cls):
    attempts = [
        lambda: cls(STORE_DIR),
        lambda: cls(str(STORE_DIR)),
        lambda: cls(root=STORE_DIR),
        lambda: cls(root=str(STORE_DIR)),
        lambda: cls(store_dir=STORE_DIR),
        lambda: cls(store_dir=str(STORE_DIR)),
        lambda: cls(data_dir=STORE_DIR),
        lambda: cls(data_dir=str(STORE_DIR)),
    ]
    errors = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    print("[debug] Store constructor attempts failed:")
    for err in errors:
        print(f"  - {err}")
    sig = inspect.signature(cls)
    fail(f"Could not construct store {cls} with STORE_DIR={STORE_DIR}. Signature: {sig}")


def default_crop_box():
    # Standard ImageNet val path for a 256 view and 224 center crop.
    if CROP_BOX_ENV:
        vals = [int(x.strip()) for x in CROP_BOX_ENV.split(",")]
        if len(vals) != 4:
            fail("CROP_BOX must have four comma-separated ints: x1,y1,x2,y2")
        return tuple(vals)
    margin = (VIEW_SIZE - OUTPUT_SIZE) // 2
    return (margin, margin, margin + OUTPUT_SIZE, margin + OUTPUT_SIZE)


def to_chw_float01(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not torch.is_tensor(x):
        fail(f"CropDCT read returned non-tensor type: {type(x)}")
    x = x.detach().cpu()
    if x.ndim == 4:
        if x.shape[0] != 1:
            print(f"[warn] Decode returned batch shape {tuple(x.shape)}; using first item.")
        x = x[0]
    if x.ndim != 3:
        fail(f"Expected decoded tensor with shape C,H,W or 1,C,H,W, got {tuple(x.shape)}")
    if x.shape[0] != 3 and x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    if x.shape[0] != 3:
        fail(f"Expected 3 channels after conversion, got shape {tuple(x.shape)}")
    x = x.float()
    if x.max().item() > 2.0:
        print("[warn] CropDCT tensor appears to be 0..255; dividing by 255 for diagnostics.")
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def read_cropdct_rgb(store):
    crop_box = default_crop_box()
    bands = None
    if hasattr(store, "num_bands"):
        try:
            bands = list(range(int(store.num_bands)))
        except Exception:
            bands = None

    attempts = []
    if hasattr(store, "read_crop"):
        base = {"image_id": IMAGE_ID, "output_size": OUTPUT_SIZE}
        attempts.append(lambda: store.read_crop(**base, crop_box=crop_box, freq_bands=bands))
        attempts.append(lambda: store.read_crop(**base, crop_box=crop_box))
        attempts.append(lambda: store.read_crop(**base))
        attempts.append(lambda: store.read_crop(IMAGE_ID, crop_box, bands, OUTPUT_SIZE))
        attempts.append(lambda: store.read_crop(IMAGE_ID, crop_box))
        attempts.append(lambda: store.read_crop(IMAGE_ID))
    if hasattr(store, "__getitem__"):
        attempts.append(lambda: store[IMAGE_ID])

    errors = []
    for attempt in attempts:
        try:
            out = attempt()
            print(f"[info] CropDCT read succeeded with crop_box={crop_box}, output_size={OUTPUT_SIZE}")
            return to_chw_float01(out)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    print("[debug] CropDCT read attempts failed:")
    for err in errors:
        print(f"  - {err}")
    fail("Could not read a crop. Set STORE_CLASS/CROP_BOX/IMAGE_ID explicitly.")


def baseline_rgb_from_raw():
    if not RAW_IMAGE:
        return None
    path = Path(RAW_IMAGE).expanduser()
    if not path.exists():
        fail(f"RAW_IMAGE does not exist: {path}")
    transform = T.Compose([
        T.Resize(VIEW_SIZE, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(OUTPUT_SIZE),
        T.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def normalize(x):
    return (x - MEAN) / STD


def denormalize(x):
    return (x * STD + MEAN).clamp(0.0, 1.0)


def stats(name, x):
    ch_mean = x.mean(dim=(1, 2))
    ch_std = x.std(dim=(1, 2))
    print(f"\n[{name}]")
    print(f"  shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"  min={x.min().item():.6f} max={x.max().item():.6f}")
    print(f"  mean={x.mean().item():.6f} std={x.std().item():.6f}")
    print(f"  channel_mean={[round(v, 6) for v in ch_mean.tolist()]}")
    print(f"  channel_std ={[round(v, 6) for v in ch_std.tolist()]}")


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def diagnose_range(name, x):
    mn = x.min().item()
    mx = x.max().item()
    mean = x.mean().item()
    if mx <= 0.02:
        print(f"[fail] {name}: max <= 0.02, likely divided by 255 twice or nearly black.")
    elif mx > 2.0:
        print(f"[fail] {name}: max > 2 before normalization, likely still in 0..255 range.")
    elif mean < 0.05:
        print(f"[warn] {name}: very dark mean; check extra division or color conversion.")
    elif mean > 0.95:
        print(f"[warn] {name}: very bright mean; check scaling/color conversion.")
    else:
        print(f"[pass] {name}: raw RGB range looks plausible.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    add_repo_to_path()
    cls = import_store_class()
    store = make_store(cls)

    crop_rgb = read_cropdct_rgb(store)
    base_rgb = baseline_rgb_from_raw()

    stats("cropdct_rgb_0_1", crop_rgb)
    diagnose_range("cropdct_rgb_0_1", crop_rgb)
    save_image(crop_rgb, OUT_DIR / "cropdct_rgb.png")

    crop_norm = normalize(crop_rgb)
    stats("cropdct_normalized_once", crop_norm)
    save_image(denormalize(crop_norm), OUT_DIR / "cropdct_norm_denorm.png")

    if base_rgb is None:
        print("\n[info] RAW_IMAGE not set, so baseline-vs-CropDCT pixel diff was skipped.")
        print(f"[info] Wrote images to: {OUT_DIR.resolve()}")
        return

    stats("baseline_rgb_0_1", base_rgb)
    diagnose_range("baseline_rgb_0_1", base_rgb)
    save_image(base_rgb, OUT_DIR / "baseline_rgb.png")

    if base_rgb.shape != crop_rgb.shape:
        fail(f"Shape mismatch: baseline={tuple(base_rgb.shape)} cropdct={tuple(crop_rgb.shape)}")

    diff = (base_rgb - crop_rgb).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    print("\n[raw RGB diff]")
    print(f"  mean_abs_diff={mean_diff:.6f}")
    print(f"  max_abs_diff ={max_diff:.6f}")
    print(f"  psnr_db      ={psnr(base_rgb, crop_rgb):.2f}")

    save_image((diff * 10.0).clamp(0.0, 1.0), OUT_DIR / "absdiff_x10.png")

    base_norm = normalize(base_rgb)
    crop_norm = normalize(crop_rgb)
    ndiff = (base_norm - crop_norm).abs()
    print("\n[normalized diff]")
    print(f"  mean_abs_diff={ndiff.mean().item():.6f}")
    print(f"  max_abs_diff ={ndiff.max().item():.6f}")

    print("\n[diagnosis]")
    if mean_diff < 0.01:
        print("  PASS: raw baseline and CropDCT tensors are very close.")
        print("  The 16-point gap is likely in training/eval setup, not decode quality.")
    elif mean_diff < 0.05:
        print("  WARN: tensors differ, but not catastrophically.")
        print("  Check interpolation mode, crop coordinates, and color conversion.")
    else:
        print("  FAIL: tensors differ too much. This is enough to explain a large accuracy gap.")
        print("  Most likely: crop box/resize mismatch, missing/double scaling, or color conversion bug.")

    print(f"\n[info] Wrote images to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
PY

