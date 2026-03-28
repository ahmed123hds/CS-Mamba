"""
Mamba Classifier using the official mamba_ssm package (v2.3.1)
==============================================================
Uses the CUDA-optimized Mamba block from Tri Dao & Albert Gu
with the real selective scan kernel.

Architecture:
    x (B, L, d_model) → N × [Mamba Block + Residual + LayerNorm]
                       → Mean Pool → Linear → logits (B, n_classes)
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """
    Wraps the official Mamba block with residual connection and LayerNorm.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        return x + self.mamba(self.norm(x))


class MambaClassifier(nn.Module):
    """
    Stack of official Mamba blocks → mean pool → linear classifier.

    Args:
        d_model:    token embedding dimension
        n_classes:  number of output classes
        n_layers:   number of stacked Mamba blocks
        d_state:    SSM state size
    """

    def __init__(
        self,
        d_model:   int,
        n_classes: int,
        n_layers:  int = 2,
        d_state:   int = 16,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → logits (B, n_classes)"""
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)     # mean pool over sequence
        return self.head(x)
