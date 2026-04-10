"""
Continuous Spatial Mamba v3 — Learned Reaction-Diffusion
========================================================
KEY FIX: S-FREE DIMENSIONALITY 
(B,N,D,S) tensor caused 3GB+ OOM crashes on TPUs. We now use a lean
(B,N,D) hidden state for spatial processing, drastically reducing memory 
while retaining the Allen-Cahn pattern formation.

Physics: The PDE is purely spatial reaction-diffusion:
  ∂h/∂t = Force_1(Learned Diffusion) + Force_2(Allen-Cahn Reaction)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Stochastic Depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        )
        return x * random_tensor / keep_prob


def cs_mamba_forward_v3(h, delta_d, diffusion_conv,
                        reaction_alpha, reaction_beta, K, H, W):
    """
    Lean S-Free Reaction-Diffusion PDE Integration.
    Hidden state: (B, N, D) — NO S dimension to avoid OOM.

    Args:
        h:               (B, N, D) initial state
        delta_d:         (B, N, D) diffusion time scale
        diffusion_conv:  nn.Conv2d — learned spatial mixing operator
        reaction_alpha:  (1, 1, D) reaction alpha coefficient
        reaction_beta:   (1, 1, D) reaction beta coefficient
        K:               int, number of Euler steps
        H, W:            spatial grid dimensions
    """
    B_val, N, D_dim = h.shape
    dt = 1.0 / K

    for _ in range(K):
        # ── Force 1: LEARNED Spatial Diffusion ───────────────────
        # Reshape to 2D
        h_2d = h.transpose(1, 2).view(B_val, D_dim, H, W)
        h_pad = F.pad(h_2d, (1, 1, 1, 1), mode='replicate')
        
        diff_h_2d = diffusion_conv(h_pad)
        diff_h = diff_h_2d.view(B_val, D_dim, N).transpose(1, 2)
        
        force_1 = delta_d * diff_h

        # ── Force 2: Nonlinear Reaction (Allen-Cahn division-free)
        # R(h) = α·h - β·h³
        force_2 = reaction_alpha * h - reaction_beta * (h ** 3)

        # ── Explicit Euler Step ──────────────────────────────────
        h = h + dt * (force_1 + force_2)

    return h


class ContinuousSpatialSSM_V3(nn.Module):
    """
    Learned Reaction-Diffusion Spatial SSM — Lean (B, N, D) state.
    """
    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)

        # Phase gate for diffusion strength
        self.dt_diff_proj = nn.Linear(d_inner, d_inner, bias=True)
        dt_init = math.log(math.exp(0.1) - 1.0)
        nn.init.constant_(self.dt_diff_proj.bias, dt_init)
        nn.init.uniform_(self.dt_diff_proj.weight, -1e-4, 1e-4)

        self.D = nn.Parameter(torch.ones(d_inner))

        # ── LEARNED Diffusion Operator ──
        self.diffusion_conv = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=0,
            groups=d_inner, bias=False
        )
        with torch.no_grad():
            lap_init = torch.tensor(
                [[0.0, 1.0, 0.0],
                 [1.0, -4.0, 1.0],
                 [0.0, 1.0, 0.0]], dtype=torch.float32
            ).view(1, 1, 3, 3) * 0.1
            self.diffusion_conv.weight.copy_(lap_init.repeat(d_inner, 1, 1, 1))

        # ── Reaction Term (Allen-Cahn polynomial) ──
        self.reaction_alpha = nn.Parameter(torch.zeros(1, 1, d_inner))
        self.reaction_beta = nn.Parameter(torch.zeros(1, 1, d_inner))
        
        self.out_proj = nn.Linear(d_inner, d_inner, bias=False)

    def forward(self, x: torch.Tensor, K_steps: int = 2, **kwargs) -> torch.Tensor:
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N

        # Per-patch diffusion context gate
        delta_d = torch.clamp(F.softplus(self.dt_diff_proj(x)), max=0.15)

        h = x.clone()
        h = cs_mamba_forward_v3(
            h, delta_d,
            self.diffusion_conv, self.reaction_alpha, self.reaction_beta,
            K_steps, H, W
        )

        y = self.out_proj(h) + x * self.D
        return y


class ContinuousSpatialMambaBlock_V3(nn.Module):
    """Reaction-Diffusion Mamba Block. Drop-in replacement."""
    def __init__(self, d_model: int, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_model, expand * d_model * 2, bias=False)
        self.local_conv2d = nn.Conv2d(
            expand * d_model, expand * d_model,
            kernel_size=3, padding=1, groups=expand * d_model, bias=True
        )
        self.activation = nn.SiLU()
        
        self.ssm = ContinuousSpatialSSM_V3(d_model=d_model, expand=expand)
        
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(expand * d_model, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, K_steps: int = 2, **kwargs) -> torch.Tensor:
        residual = x
        B, N, D = x.shape
        H = W = int(math.sqrt(N))

        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)

        u_2d = u.transpose(1, 2).view(B, -1, H, W)
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.view(B, -1, N).transpose(1, 2)
        u = self.activation(u)

        y_ssm = self.ssm(u, K_steps=K_steps)
        out = self.out_proj(y_ssm * F.silu(z))
        
        return residual + self.drop_path(out)


class CSMamba_V3(nn.Module):
    """
    CS-Mamba V3 — Learned Reaction-Diffusion Architecture
    Lean (B,N,D) hidden state version to avoid TPU OOM
    """
    def __init__(self, cfg):
        super().__init__()
        img_size   = getattr(cfg, 'img_size', getattr(cfg, 'canvas_size', 224))
        patch_size = getattr(cfg, 'patch_size', 16)
        d_embed    = getattr(cfg, 'd_embed', 384)
        n_layers   = getattr(cfg, 'n_mamba_layers', 12)
        n_classes  = getattr(cfg, 'n_classes', 1000)
        K_steps    = getattr(cfg, 'K_steps', 2)
        drop_path_rate = getattr(cfg, 'drop_path', 0.1)

        self.K_steps = K_steps
        num_patches  = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([
            ContinuousSpatialMambaBlock_V3(d_model=d_embed, drop_path=dp_rates[i])
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"CSMamba_V3 initialized: {n_params/1e6:.1f}M params, "
                    f"d={d_embed}, layers={n_layers}, K={K_steps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed

        for layer in self.layers:
            x = layer(x, K_steps=self.K_steps)

        x = self.final_norm(x)
        return self.head(x.mean(dim=1))
