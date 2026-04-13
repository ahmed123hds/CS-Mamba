"""
Continuous Spatial Mamba v4 — Complex Schrödinger Reaction-Diffusion
====================================================================
KEY FIX over previous broken versions:
  The hidden state is (B, N, D) NOT (B, N, D, S).
  The (B,N,D,S) tensor caused 3+ GB per tensor → TPU OOM → BrokenProcessPool.
  Complex (real, imag) ALREADY doubles expressivity — no S dimension needed.

Physics:
  Heat Eq  (V2):  ∂h/∂t = D∇²h              → dissipative, blurs features
  Schrödinger (V4): ∂ψ/∂t = i·D·∇²ψ + R(ψ)  → unitary + Ginzburg-Landau

ψ = h_real + i·h_imag  (two (B,N,D) tensors — same size as V1/V2 inputs)
i * laplacian(ψ) = [-laplacian(h_imag), +laplacian(h_real)]

Memory per tensor: B × N × D = 128 × 256 × 384 = 12M floats = 48MB ✓
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


def cs_mamba_forward_v4(h_real, h_imag, delta_d, diffusion_conv,
                         reaction_alpha, reaction_beta, K, H, W):
    """
    Lean Complex Schrödinger PDE Integration.
    Hidden state: (B, N, D) — NO S dimension to avoid OOM.

    Schrödinger cross-coupling:
      ∂h_real/∂t = -D·∇²h_imag    (imaginary feeds real)
      ∂h_imag/∂t = +D·∇²h_real    (real feeds imaginary)

    Reaction (Ginzburg-Landau, division-free):
      R(ψ) = ψ·(α - β|ψ|²)
    """
    B_val, N, D_dim = h_real.shape
    dt = 1.0 / K

    for _ in range(K):
        # ── Learned spatial operator on REAL part → feeds IMAG ────
        h_r_2d = h_real.transpose(1, 2).view(B_val, D_dim, H, W)
        h_r_pad = F.pad(h_r_2d, (1, 1, 1, 1), mode='constant', value=0.0)
        lap_real = diffusion_conv(h_r_pad)          # (B, D, H, W)
        lap_real = lap_real.view(B_val, D_dim, N).transpose(1, 2)  # (B, N, D)

        # ── Learned spatial operator on IMAG part → feeds REAL ────
        h_i_2d = h_imag.transpose(1, 2).view(B_val, D_dim, H, W)
        h_i_pad = F.pad(h_i_2d, (1, 1, 1, 1), mode='constant', value=0.0)
        lap_imag = diffusion_conv(h_i_pad)
        lap_imag = lap_imag.view(B_val, D_dim, N).transpose(1, 2)

        # Schrödinger: i * lap(ψ) = [-lap(imag), +lap(real)]
        f2_real = delta_d * (-lap_imag)   # imag → real
        f2_imag = delta_d * (+lap_real)   # real → imag

        # ── Ginzburg-Landau reaction (completely division-free!) ───
        mag_sq = h_real ** 2 + h_imag ** 2           # (B, N, D)
        scale = reaction_alpha - reaction_beta * mag_sq
        f3_real = h_real * scale
        f3_imag = h_imag * scale

        # ── Euler step ────────────────────────────────────────────
        h_real = h_real + dt * (f2_real + f3_real)
        h_imag = h_imag + dt * (f2_imag + f3_imag)

    return h_real, h_imag


class ContinuousSpatialSSM_V4(nn.Module):
    """
    Complex Schrödinger SSM — lean (B, N, D) hidden state.
    """
    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)

        # Phase gate: learns how much each patch participates in diffusion
        self.dt_diff_proj = nn.Linear(d_inner, d_inner, bias=True)
        dt_init = math.log(math.exp(0.1) - 1.0)
        nn.init.constant_(self.dt_diff_proj.bias, dt_init)
        nn.init.uniform_(self.dt_diff_proj.weight, -1e-4, 1e-4)

        # Skip connection scale
        self.D = nn.Parameter(torch.ones(d_inner))

        # ── Learned diffusion conv (same as V3, shared real/imag) ──
        self.diffusion_conv = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=0,
            groups=d_inner, bias=False
        )
        with torch.no_grad():
            lap_init = torch.tensor(
                [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32
            ).view(1, 1, 3, 3) * 0.1
            self.diffusion_conv.weight.copy_(lap_init.repeat(d_inner, 1, 1, 1))

        # ── Ginzburg-Landau reaction parameters (channel-wise) ──
        self.reaction_alpha = nn.Parameter(torch.zeros(1, 1, d_inner))
        self.reaction_beta  = nn.Parameter(torch.zeros(1, 1, d_inner))

        # Output: project complex (real+imag) → d_inner
        self.out_complex = nn.Linear(d_inner * 2, d_inner, bias=False)

    def forward(self, x: torch.Tensor, K_steps: int = 2, **kwargs) -> torch.Tensor:
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N

        # Per-patch diffusion gate (scalar weight on diffusion strength)
        delta_d = torch.clamp(F.softplus(self.dt_diff_proj(x)), max=0.15)

        # Initial complex state: real from input, imaginary starts at 0
        h_real = x.clone()
        h_imag = torch.zeros_like(x)

        h_real, h_imag = cs_mamba_forward_v4(
            h_real, h_imag, delta_d,
            self.diffusion_conv, self.reaction_alpha, self.reaction_beta,
            K_steps, H, W
        )

        # Combine real + imag via learned projection, then skip
        combined = torch.cat([h_real, h_imag], dim=-1)   # (B, N, 2D)
        y = self.out_complex(combined) + x * self.D
        return y


class ContinuousSpatialMambaBlock_V4(nn.Module):
    """Complex Schrödinger Mamba Block. Drop-in replacement for V1/V2/V3."""
    def __init__(self, d_model: int, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, expand * d_model * 2, bias=False)

        self.local_conv2d = nn.Conv2d(
            expand * d_model, expand * d_model,
            kernel_size=3, padding=1, groups=expand * d_model, bias=True
        )
        self.activation = nn.SiLU()
        self.ssm = ContinuousSpatialSSM_V4(d_model=d_model, expand=expand)
        self.out_proj = nn.Linear(expand * d_model, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, K_steps: int = 2, **kwargs) -> torch.Tensor:
        residual = x
        B, N, D = x.shape
        H = W = int(math.sqrt(N))

        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)

        u_2d = u.transpose(1, 2).view(B, -1, H, W)
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.view(B, -1, N).transpose(1, 2)
        u = self.activation(u)

        y_ssm = self.ssm(u, K_steps=K_steps)
        out = self.out_proj(y_ssm * F.silu(z))
        return residual + self.drop_path(out)


class CSMamba_V4(nn.Module):
    """
    CS-Mamba V4 — Complex Schrödinger Reaction-Diffusion
    =====================================================
    Hidden state: (B, N, D) — lean, no S blowup.
    Wave function ψ = h_real + i·h_imag evolves via Schrödinger + Ginzburg-Landau.
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
            ContinuousSpatialMambaBlock_V4(d_model=d_embed, drop_path=dp_rates[i])
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"CSMamba_V4 (Complex Schrödinger) {n_params/1e6:.1f}M params, "
                    f"d={d_embed}, layers={n_layers}, K={K_steps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            x = layer(x, K_steps=self.K_steps)
        return self.head(self.final_norm(x).mean(dim=1))
