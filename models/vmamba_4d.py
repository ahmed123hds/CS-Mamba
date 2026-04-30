"""
TPU-safe VMamba-style 4-direction Cross-Scan baseline.

This is a lightweight benchmark model inspired by VMamba's SS2D idea:
scan a 2D feature map along left/right and top/bottom routes, run a selective
recurrent state update on each route, then cross-merge back to the image grid.

It intentionally avoids CUDA/Triton selective-scan kernels so it can run under
PyTorch/XLA on TPU.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * mask / keep_prob


class CrossScanSS2D(nn.Module):
    """
    Four-route axis-wise selective scan over a square image-token grid.

    Routes:
      1. left-to-right over each row
      2. right-to-left over each row
      3. top-to-bottom over each column
      4. bottom-to-top over each column

    The official VMamba-style flattened cross-scan traces length H*W sequences.
    On TPU/XLA that creates a very large recurrent graph before step 0. This
    axis-wise variant keeps the recurrent scan but reduces the traced length to
    max(H, W), which is the TPU-safe 2D benchmark path.
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.n_routes = 4

        self.delta_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.value_proj = nn.Linear(d_inner, d_inner, bias=False)
        self.route_scale = nn.Parameter(torch.ones(self.n_routes, 1, 1, d_inner))
        self.out_norm = nn.LayerNorm(d_inner)
        self.D = nn.Parameter(torch.ones(d_inner))

        # softplus(-1.5) ~= 0.20, exp(-0.20) ~= 0.82 retain.
        nn.init.constant_(self.delta_proj.bias, -1.5)

    def _selective_scan(self, seq: torch.Tensor, route_idx: int) -> torch.Tensor:
        """
        seq: (B, L, D)
        Returns: (B, L, D)
        """
        delta = F.softplus(self.delta_proj(seq))
        retain = torch.exp(-delta)
        inject = 1.0 - retain
        values = self.value_proj(seq) * self.route_scale[route_idx]

        states = []
        h = torch.zeros_like(values[:, 0])
        for t in range(values.shape[1]):
            h = retain[:, t] * h + inject[:, t] * values[:, t]
            states.append(h)
        return torch.stack(states, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D), where N must be a square grid.
        """
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "VMamba4D requires a square token grid."

        grid = x.transpose(1, 2).reshape(B, D, H, W)

        rows = grid.permute(0, 2, 3, 1).reshape(B * H, W, D)
        cols = grid.permute(0, 3, 2, 1).reshape(B * W, H, D)

        y_lr = self._selective_scan(rows, 0).reshape(B, H, W, D)
        y_rl = torch.flip(
            self._selective_scan(torch.flip(rows, dims=[1]), 1),
            dims=[1],
        ).reshape(B, H, W, D)

        y_tb = self._selective_scan(cols, 2).reshape(B, W, H, D).permute(0, 2, 1, 3)
        y_bt = torch.flip(
            self._selective_scan(torch.flip(cols, dims=[1]), 3),
            dims=[1],
        ).reshape(B, W, H, D).permute(0, 2, 1, 3)

        merged = (y_lr + y_rl + y_tb + y_bt) * 0.25
        merged = merged.reshape(B, N, D)
        return self.out_norm(merged) + x * self.D


class VMamba4DBlock(nn.Module):
    """VMamba-style VSS block with local depthwise conv + four-route SS2D."""

    def __init__(self, d_model: int, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        d_inner = int(expand * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.local_conv2d = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True
        )
        self.activation = nn.SiLU()
        self.ss2d = CrossScanSS2D(d_model=d_model, expand=expand)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, N, _ = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "VMamba4D requires a square token grid."

        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)

        u_2d = u.transpose(1, 2).reshape(B, -1, H, W)
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.flatten(2).transpose(1, 2)
        u = self.activation(u)

        y = self.ss2d(u)
        out = self.out_proj(y * F.silu(z))
        return residual + self.drop_path(out)


class VMamba4D(nn.Module):
    """
    TinyImageNet-friendly TPU-safe VMamba-style classifier.

    This keeps the same patch embedding / pooling style as CSMamba_V6 so the
    benchmark mainly tests fixed 2D row/column scan transport versus learned
    characteristic transport.
    """

    def __init__(self, cfg):
        super().__init__()
        img_size = getattr(cfg, "img_size", getattr(cfg, "canvas_size", 224))
        patch_size = getattr(cfg, "patch_size", 16)
        d_embed = getattr(cfg, "d_embed", 384)
        n_layers = getattr(cfg, "n_mamba_layers", 12)
        n_classes = getattr(cfg, "n_classes", 1000)
        drop_path_rate = getattr(cfg, "drop_path", 0.1)

        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([
            VMamba4DBlock(d_model=d_embed, expand=2, drop_path=dp_rates[i])
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "VMamba4D %.1fM params, d=%d, layers=%d",
            n_params / 1e6, d_embed, n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.head(self.final_norm(x).mean(dim=1))


__all__ = ["CrossScanSS2D", "VMamba4DBlock", "VMamba4D"]
