"""
Patch Encoder
=============
Splits a CIFAR-10 image into non-overlapping patches and embeds
each patch into a d_embed-dimensional vector.

CIFAR-10 image: 32×32×3
With patch_size=8: 4×4 = 16 patches of size 8×8×3

Architecture per patch:
    Flatten(8×8×3 = 192 dims)
        → Linear(192, d_embed)
        → LayerNorm
        → GELU

Optionally: use a shallow CNN per patch instead of flatten+linear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Splits image into patches and projects each patch to d_embed dims.

    Args:
        img_size:    image height/width (assumes square)
        patch_size:  patch height/width (assumes square)
        in_channels: number of image channels
        d_embed:     output embedding dimension
    """

    def __init__(
        self,
        img_size:    int = 32,
        patch_size:  int = 8,
        in_channels: int = 3,
        d_embed:     int = 128,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.patch_size  = patch_size
        self.grid_h      = img_size // patch_size   # number of patch rows
        self.grid_w      = img_size // patch_size   # number of patch cols
        self.n_patches   = self.grid_h * self.grid_w
        self.patch_dim   = patch_size * patch_size * in_channels

        # Simple linear projection (like ViT)
        self.projection = nn.Sequential(
            nn.Linear(self.patch_dim, d_embed),
            nn.LayerNorm(d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            embeddings: (B, n_patches, d_embed)
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Rearrange into patches: (B, n_patches, patch_dim)
        # unfold along height then width
        x = x.unfold(2, p, p).unfold(3, p, p)
        # x: (B, C, grid_h, grid_w, p, p)
        x = x.contiguous().view(B, C, self.grid_h, self.grid_w, p * p)
        # (B, grid_h, grid_w, C, p*p)
        x = x.permute(0, 2, 3, 1, 4)
        # (B, n_patches, C*p*p)
        x = x.reshape(B, self.n_patches, -1)

        return self.projection(x)    # (B, n_patches, d_embed)
