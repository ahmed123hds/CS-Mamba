"""
CS-Mamba: Spatial Sensitivity Map
==================================
Jacobian-based spatial influence measurement (no training required).

Method:
  For each spatial patch position (i,j):
    1. Take a baseline input x
    2. Add +epsilon to patch (i,j)
    3. Measure change in center patch output
    4. Record change magnitude as the "influence" of patch (i,j) on center

This directly proves which patches can "see" the center patch and vice-versa.
  - Ideal result: V4 shows a full, smooth radial glow (global context)
  - Expected V1  : harsh star/cross pattern (1D scan artifacts)
  - Expected V3  : fading circle (real diffusion: dissipates energy)
"""

import sys, math, os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba_V1
from models.continuous_spatial_mamba_v3 import CSMamba_V3
from models.continuous_spatial_mamba_v4 import CSMamba_V4


# ─────────────────────────────────────────────────
class Cfg:
    img_size = 64; canvas_size = 64
    patch_size = 4            # → 16×16 = 256 patches
    d_embed = 64; d_state = 8
    n_mamba_layers = 4
    K_steps = 6               # more steps = more PDE propagation visible
    n_classes = 200
    drop_path = 0.0
# ─────────────────────────────────────────────────


def get_ssm_block(model):
    """Return first SSM block for targeted testing."""
    if hasattr(model, 'layers'):   return model.layers[0]   # V4
    if hasattr(model, 'blocks'):   return model.blocks[0]   # V1/V3
    raise AttributeError("Cannot find SSM block")


@torch.no_grad()
def jacobian_sensitivity(model, device, eps=1.0):
    """
    Directly tests the core SSM block (first layer only) to isolate
    the ARCHITECTURAL effect of V1 vs V3 vs V4 purely from initialization.

    Input:  random patch sequence (B=1, N, D)
    Target: center patch output (N//2)
    Perturbation: nudge each patch embedding by +eps along its mean direction
    """
    model = model.to(device).eval()

    cfg = Cfg()
    H = W = cfg.img_size // cfg.patch_size  # = 16
    N = H * W                               # = 256
    D = cfg.d_embed
    center_idx = (H // 2) * W + (W // 2)

    # Standard baseline sequence
    torch.manual_seed(42)
    x_base = torch.randn(1, N, D, device=device) * 0.1  # small, stable

    # Get baseline output for center patch
    block = get_ssm_block(model)
    out_base = block(x_base, K_steps=cfg.K_steps) if hasattr(block, 'K_steps') or True else block(x_base)
    center_base = out_base[0, center_idx, :]    # (D,)

    sensitivity = np.zeros(N)

    for patch_idx in range(N):
        x_pert = x_base.clone()
        # Perturb all channels of patch_idx
        x_pert[0, patch_idx, :] += eps

        out_pert = block(x_pert, K_steps=cfg.K_steps) if True else block(x_pert)
        center_pert = out_pert[0, center_idx, :]

        diff = (center_pert - center_base).abs().sum().item()
        sensitivity[patch_idx] = diff

    # Reshape to 2D spatial grid and power-scale to boost contrast
    sens_map = sensitivity.reshape(H, W)
    vmax = sens_map.max()
    if vmax > 0:
        sens_map = (sens_map / vmax) ** 0.3   # power < 1 boosts low values visually

    return sens_map, H, W, center_idx


def make_colormap_figure(maps_data, titles, save_path):
    """Render paper-quality 3-panel figure."""
    fig = plt.figure(figsize=(18, 7), facecolor='#0d0d0d')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06)

    cmaps = ['magma', 'inferno', 'plasma']
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']

    for col, (entry, title, cmap, accent) in enumerate(
        zip(maps_data, titles, cmaps, colors)
    ):
        sens, H, W, cidx = entry  # unpack tuple correctly
        ax = fig.add_subplot(gs[col])
        ax.set_facecolor('black')

        im = ax.imshow(sens, cmap=cmap, vmin=0, vmax=1,
                       interpolation='bicubic', aspect='equal')

        # Center cross-hair
        cy, cx = cidx // W, cidx % W
        ax.plot([cx], [cy], 'w+', markersize=18, markeredgewidth=2.5, alpha=0.9)
        ax.plot([cx], [cy], 'o', markersize=6, color=accent,
                markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(2.5)

        ax.set_title(title, color='white', fontsize=13, pad=12,
                     fontfamily='monospace', linespacing=1.5)

        # Percentage "global influence" metric
        influenced = np.mean(sens > 0.001) * 100
        global_mean = sens.mean() * 100
        ax.text(0.98, 0.03, f"Global reach: {influenced:.0f}%  |  mean: {global_mean:.1f}%",
                transform=ax.transAxes, color=accent, fontsize=9.5,
                ha='right', va='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    fig.suptitle(
        "CS-Mamba — Spatial Influence Map\n"
        "(How much does perturbing each patch change the center patch's output?)",
        color='white', fontsize=14, y=1.02, fontfamily='monospace'
    )

    plt.savefig(save_path, dpi=250, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    print(f"✅ Saved: {save_path}")
    return fig


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    cfg = Cfg()
    models = [
        (CSMamba_V1,  "V1  |  1D Bidirectional Scan\n(Flat sequence → breaks spatial symmetry)"),
        (CSMamba_V3,  "V3  |  Real Heat Equation\n(∂h/∂t = D∇²h  →  energy dissipates)"),
        (CSMamba_V4,  "V4  |  Complex Schrödinger  [Ours]\n(∂ψ/∂t = iD∇²ψ + R(ψ)  →  energy conserved)"),
    ]

    maps_data = []
    titles = []

    for cls, title in models:
        name = title.split('|')[0].strip()
        print(f"Computing Jacobian sensitivity for {name}...")
        model = cls(cfg)
        sens, H, W, cidx = jacobian_sensitivity(model, device)
        maps_data.append((sens, H, W, cidx))
        titles.append(title)
        print(f"  {name}: mean={sens.mean():.4f}  max={sens.max():.4f}  "
              f"coverage(>5%)={np.mean(sens>0.05)*100:.1f}%")

    save_path = 'spatial_sensitivity_v1_v3_v4.png'
    make_colormap_figure(maps_data, titles, save_path)


if __name__ == '__main__':
    main()
