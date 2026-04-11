import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from models.continuous_spatial_mamba import ContinuousSpatialMambaClassifier as CSMamba_V1
from models.continuous_spatial_mamba_v3 import CSMamba_V3
from models.continuous_spatial_mamba_v4 import CSMamba_V4

class EmptyConfig:
    pass

def get_config():
    cfg = EmptyConfig()
    cfg.img_size = 224
    cfg.patch_size = 16
    cfg.d_embed = 192
    cfg.n_mamba_layers = 12
    cfg.K_steps = 3
    cfg.n_classes = 1000
    cfg.canvas_size = 224
    cfg.drop_path = 0.0
    return cfg

def compute_erf(model_class, device='cuda'):
    cfg = get_config()
    model = model_class(cfg).to(device)
    model.eval()
    
    # Try to find the component right before global pooling
    target_module = None
    if hasattr(model, 'final_norm'):
        target_module = model.final_norm
    elif hasattr(model, 'norm_f'):
        target_module = model.norm_f
    else:
        # Fallback to the last layer
        target_module = model.layers[-1]

    features = []
    def hook(m, i, o):
        # Depending on the module, 'o' might be a tuple. Mamba standard normally returns raw tensor.
        if isinstance(o, tuple):
            features.append(o[0])
        else:
            features.append(o)

    handle = target_module.register_forward_hook(hook)

    # Blank image, requires_grad=True
    x = torch.zeros(1, 3, 224, 224, requires_grad=True, device=device)
    
    # Forward pass
    model(x)
    
    feat = features[0] 
    # Mamba output: (B, N, D)
    B, N, D = feat.shape
    H = W = int(math.sqrt(N))
    
    # Find the absolute center patch
    center_idx = (H // 2) * W + (W // 2)
    
    # Compute the gradient of the center patch vector w.r.t to the input image
    loss = feat[0, center_idx, :].sum()
    loss.backward()
    
    # ERF is the magnitude of the gradient on the spatial image dims
    grad = x.grad[0].abs().sum(dim=0).cpu().numpy()  # (224, 224)
    
    # Apply a non-linear scaling to make the heatmap visually pop for papers
    grad = grad ** 0.5 
    grad = grad / (grad.max() + 1e-10)
    
    handle.remove()
    return grad

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Generating Effective Receptive Field (ERF) heatmaps on {device}...")
    
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    models = [
        (CSMamba_V1, "V1: 1D Bidirectional Scan\n(Star-pattern tearing)"),
        (CSMamba_V3, "V3: Heat Equation\n(Blurred diffusion)"),
        (CSMamba_V4, "V4: Complex Schrödinger\n(Ours: Global unitary context)")
    ]
    
    for ax, (cls, title) in zip(axes, models):
        print(f"Tracing {title.split(':')[0]}...")
        
        try:
            erf = compute_erf(cls, device)
            
            # We use 'magma' because it beautifully highlights low-intensity gradients
            sns.heatmap(erf, ax=ax, cmap='magma', cbar=False, square=True, 
                        xticklabels=False, yticklabels=False)
            
            ax.set_title(title, fontsize=14, pad=15, linespacing=1.4)
            
            # Subtle targeting reticle pointing exactly at the center patch
            ax.plot([112, 112], [105, 119], color='white', alpha=0.5, lw=1.0)
            ax.plot([105, 119], [112, 112], color='white', alpha=0.5, lw=1.0)
            
        except Exception as e:
            print(f"Failed to trace {title}: {e}")
            ax.set_title(f"{title}\n[Error generating ERF]", color='red')

    plt.tight_layout()
    save_path = 'erf_ablation_v1_v3_v4.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Heatmaps saved to: {save_path}")

if __name__ == "__main__":
    main()
