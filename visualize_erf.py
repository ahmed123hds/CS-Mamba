"""
Effective Receptive Field (ERF) Visualization
==============================================
Computes the gradient of the center patch's output with respect to
all input pixels. This shows how information spatially spreads through
the model.

For CS-Mamba: Should show a smooth 2D circular/diamond spread.
For 1D Mamba: Would show a "snake" pattern following the scan path.

Usage:
    python visualize_erf.py --model v2 --checkpoint best_CSMamba_V2.pt
    python visualize_erf.py --model v1 --checkpoint best_CSMamba_V1.pt
"""

import math, argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba_V1
from models.continuous_spatial_mamba_v2 import CSMamba_V2


class Config:
    img_size = 64
    patch_size = 4
    n_classes = 200
    d_embed = 192
    d_state = 16
    n_mamba_layers = 8
    K_steps = 3
    canvas_size = 64


def compute_erf(model, device, img_size=64, n_samples=50):
    """
    Compute the Effective Receptive Field by averaging input gradients
    across multiple random images, with respect to the center patch output.
    """
    model.eval()
    grid_size = img_size // 4  # patch_size=4 → 16x16 grid
    center_patch = (grid_size * grid_size) // 2  # center of the patch grid

    accumulated_grad = torch.zeros(1, 3, img_size, img_size).to(device)

    for i in range(n_samples):
        # Random input image
        x = torch.randn(1, 3, img_size, img_size, device=device, requires_grad=True)

        # Forward pass — get the feature at the center patch
        embeddings = model.embedder(x)  # (1, N, d_embed)
        for block in model.blocks:
            embeddings = block(embeddings, K_steps=model.K_steps)
        embeddings = model.norm(embeddings)

        # Take the center patch's feature and sum it (scalar target for grad)
        center_feature = embeddings[:, center_patch, :].sum()

        # Backward to get gradient w.r.t. input pixels
        center_feature.backward()

        accumulated_grad += x.grad.abs().detach()
        x.grad = None

    # Average and collapse channels
    erf_map = accumulated_grad.squeeze(0).mean(dim=0).cpu().numpy()  # (H, W)
    erf_map /= n_samples

    return erf_map


def compute_erf_per_k(model, device, K_values=[1, 2, 3, 5], img_size=64, n_samples=30):
    """Compute ERF for different K_steps to show how receptive field grows."""
    erf_maps = {}
    model.eval()
    grid_size = img_size // 4
    center_patch = (grid_size * grid_size) // 2

    for K in K_values:
        accumulated_grad = torch.zeros(1, 3, img_size, img_size).to(device)

        for i in range(n_samples):
            x = torch.randn(1, 3, img_size, img_size, device=device, requires_grad=True)
            embeddings = model.embedder(x)
            for block in model.blocks:
                embeddings = block(embeddings, K_steps=K)
            embeddings = model.norm(embeddings)
            center_feature = embeddings[:, center_patch, :].sum()
            center_feature.backward()
            accumulated_grad += x.grad.abs().detach()
            x.grad = None

        erf_map = accumulated_grad.squeeze(0).mean(dim=0).cpu().numpy()
        erf_map /= n_samples
        erf_maps[K] = erf_map
        print(f"  ✓ Computed ERF for K={K}")

    return erf_maps


def plot_erf_single(erf_map, title, save_path):
    """Plot a single ERF heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(erf_map, cmap='inferno', interpolation='bilinear')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')

    # Mark center
    center = erf_map.shape[0] // 2
    ax.plot(center, center, 'w+', markersize=15, markeredgewidth=2)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Gradient Magnitude')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_erf_k_comparison(erf_maps, save_path):
    """Plot ERF heatmaps for different K values side by side."""
    K_values = sorted(erf_maps.keys())
    n = len(K_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, K in zip(axes, K_values):
        erf = erf_maps[K]
        im = ax.imshow(erf, cmap='inferno', interpolation='bilinear')
        ax.set_title(f'K = {K}', fontsize=14, fontweight='bold')

        center = erf.shape[0] // 2
        ax.plot(center, center, 'w+', markersize=12, markeredgewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Effective Receptive Field Growth with K Steps',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_diffusivity(model, save_path):
    """Visualize the learned diffusivity values per channel."""
    D_raw = model.blocks[0].continuous_ssm.diffusivity_raw.detach().cpu()
    D_phys = torch.sigmoid(D_raw) * 0.5
    D_values = D_phys.squeeze().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    colors = plt.cm.coolwarm(D_values / D_values.max())
    ax.bar(range(len(D_values)), D_values, color=colors, width=1.0)
    ax.set_xlabel('Channel Index', fontsize=12)
    ax.set_ylabel('Learned Diffusivity (D)', fontsize=12)
    ax.set_title('Learned Diffusivity Constants per Channel (Layer 0)',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='CFL Boundary (D=0.25)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser("ERF Visualization for CS-Mamba")
    parser.add_argument('--model', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--plot_k', action='store_true',
                        help='Plot ERF for different K values')
    parser.add_argument('--plot_diffusivity', action='store_true',
                        help='Plot learned diffusivity constants')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Config()

    # Build model
    if args.model == 'v2':
        model = CSMamba_V2(cfg).to(device)
        model_name = 'CSMamba_V2'
    else:
        model = CSMamba_V1(cfg).to(device)
        model_name = 'CSMamba_V1'

    # Load checkpoint if provided
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {model_name} | Params: {n_params:,}")
    print(f"  Device: {device}\n")

    # 1. Single ERF Map
    print("  Computing Effective Receptive Field...")
    erf_map = compute_erf(model, device, n_samples=args.n_samples)
    plot_erf_single(erf_map, f'{model_name} — Effective Receptive Field',
                    f'erf_{model_name}.png')

    # 2. ERF across K values
    if args.plot_k:
        print("\n  Computing ERF for K=1,2,3,5...")
        erf_maps = compute_erf_per_k(model, device, K_values=[1, 2, 3, 5],
                                     n_samples=args.n_samples)
        plot_erf_k_comparison(erf_maps, f'erf_k_comparison_{model_name}.png')

    # 3. Diffusivity Analysis
    if args.plot_diffusivity and args.checkpoint:
        print("\n  Plotting learned diffusivity...")
        plot_diffusivity(model, f'diffusivity_{model_name}.png')

    print("\n  ✓ All visualizations complete!")


if __name__ == '__main__':
    main()
