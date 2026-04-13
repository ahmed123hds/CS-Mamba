import math
import logging
import torch
import torch.nn as nn

from .continuous_spatial_mamba_v4 import ContinuousSpatialMambaBlock_V4
from .characteristic_mamba_v6 import CharacteristicMambaBlock_V6

logger = logging.getLogger(__name__)

class CSMamba_V6_1(nn.Module):
    """
    CS-Mamba V6.1 — Stagewise Hybrid (V4 + V6)
    Combines the fast, stable representation learning of V4 (Complex Schrödinger) in the 
    early and middle layers with the advanced, dynamic flow routing of V6 in the final layers.
    """
    def __init__(self, cfg):
        super().__init__()
        img_size = getattr(cfg, "img_size", getattr(cfg, "canvas_size", 224))
        patch_size = getattr(cfg, "patch_size", 16)
        d_embed = getattr(cfg, "d_embed", 256)
        n_layers = getattr(cfg, "n_mamba_layers", 12)
        n_classes = getattr(cfg, "n_classes", 1000)
        k_steps = getattr(cfg, "K_steps", 3)
        drop_path_rate = getattr(cfg, "drop_path", 0.15)
        n_flow_groups = getattr(cfg, "n_flow_groups", 8)
        
        # Stagewise config: fallback to 2 V6 layers if not explicitly provided
        n_v6_layers = getattr(cfg, "n_v6_layers", 2)
        n_v4_layers = max(0, n_layers - n_v6_layers)

        self.k_steps = k_steps
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        
        layers = []
        # Early layers: V4
        for i in range(n_v4_layers):
            layers.append(ContinuousSpatialMambaBlock_V4(
                d_model=d_embed, drop_path=dp_rates[i], expand=2
            ))
        # Late layers: V6
        for i in range(n_v4_layers, n_layers):
            layers.append(CharacteristicMambaBlock_V6(
                d_model=d_embed, drop_path=dp_rates[i], expand=2, n_flow_groups=n_flow_groups
            ))
            
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CSMamba_V6.1 (Hybrid) %.1fM params, d=%d, layers=%d (V4:%d, V6:%d), K=%d, G=%d",
            n_params / 1e6, d_embed, n_layers, n_v4_layers, n_v6_layers, k_steps, n_flow_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            if isinstance(layer, CharacteristicMambaBlock_V6):
                x = layer(x, k_steps=self.k_steps)
            else:
                x = layer(x, K_steps=self.k_steps)
        return self.head(self.final_norm(x).mean(dim=1))

__all__ = ["CSMamba_V6_1"]
