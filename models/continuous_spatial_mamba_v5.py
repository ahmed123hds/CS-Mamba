import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core PDE Block — fully real-valued, no torch.fft, no torch.complex,
# no reshape/view (uses only permute + flatten + unflatten).
# Laplacian applied via fixed depthwise conv (well-tested XLA autograd).
# ---------------------------------------------------------------------------


def _build_laplacian_kernel(d_inner: int) -> torch.Tensor:
    """5-point periodic Laplacian as a depthwise 3×3 kernel."""
    k = torch.tensor([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=torch.float32)
    return k.unsqueeze(0).unsqueeze(0).expand(d_inner, 1, 3, 3).contiguous()


class RealSymplecticReactionDiffusion2D(nn.Module):
    """
    TPU-native real-valued Schrödinger-inspired spatial block.

    • State: ψ = u + iv represented as two real tensors (u, v).
    • Diffusion: symplectic leapfrog with a fixed depthwise-conv Laplacian.
    • Reaction:  forward Euler on  ψ_t = (α − β|ψ|²)ψ  (no exp/sqrt).
    • Splitting: Strang  (half-reaction → full-diffusion → half-reaction).

    Every op is a standard real-valued tensor op with well-tested XLA autograd.
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        # Adaptive diffusion strength per channel
        self.diffusion_gate = nn.Linear(d_inner, d_inner, bias=True)
        nn.init.zeros_(self.diffusion_gate.weight)
        nn.init.constant_(self.diffusion_gate.bias, -3.0)      # start small
        self.max_diffusion = 0.05

        # Reaction parameters (channel-wise)
        self.reaction_alpha = nn.Parameter(torch.zeros(1, d_inner, 1, 1))
        self.reaction_beta = nn.Parameter(torch.full((1, d_inner, 1, 1), 0.1))

        # Fixed (non-learnable) Laplacian kernel — registered as buffer
        self.register_buffer("lap_kernel", _build_laplacian_kernel(d_inner))

        # Output mixer + skip
        self.out_mix = nn.Linear(d_inner * 2, d_inner, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

    # ---- spatial Laplacian via depthwise conv (periodic BC) ---------------

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 5-point Laplacian with circular (periodic) padding."""
        x_pad = F.pad(x, [1, 1, 1, 1], mode="circular")
        return F.conv2d(x_pad, self.lap_kernel, groups=self.d_inner)

    # ---- reaction (forward-Euler) -----------------------------------------

    def _reaction_step(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> tuple:
        """
        Euler step for  ψ_t = (α − β|ψ|²)ψ.
        Rate is hard-clamped to [-1, 1] so no single step can blow up.
        """
        alpha = 0.2 * torch.tanh(self.reaction_alpha)
        beta = F.softplus(self.reaction_beta).clamp(min=1e-4)
        q = u * u + v * v
        rate = (alpha - beta * q).clamp(-1.0, 1.0)
        u = u + dt * rate * u
        v = v + dt * rate * v
        return u, v

    # ---- Hamiltonian diffusion (leapfrog / Störmer-Verlet) ----------------

    def _hamiltonian_step(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
    ) -> tuple:
        """
        Symplectic leapfrog for  u_t = γ∇²v,  v_t = −γ∇²u.
        Three Laplacian evals per step ensure second-order accuracy.
        """
        v_half = v - 0.5 * dt * gamma * self._laplacian(u)
        u_new = u + dt * gamma * self._laplacian(v_half)
        v_new = v_half - 0.5 * dt * gamma * self._laplacian(u_new)
        return u_new, v_new

    # ---- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor, k_steps: int = 4) -> torch.Tensor:
        bsz, num_tokens, d_inner = x.shape
        h = w = int(math.sqrt(num_tokens))

        # Adaptive diffusion coefficient  (B, D, 1, 1)
        pooled = x.mean(dim=1)                                       # (B, D)
        gamma = F.softplus(self.diffusion_gate(pooled))
        gamma = gamma.clamp(max=self.max_diffusion)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)                     # (B, D, 1, 1)

        dt = 1.0 / max(int(k_steps), 1)

        # (B, N, D) → (B, D, N) → (B, D, H, W)  — NO reshape / view
        u = x.permute(0, 2, 1).unflatten(2, (h, w))                  # (B, D, H, W)
        v = torch.zeros_like(u)

        half_dt = 0.5 * dt
        for _ in range(int(k_steps)):
            u, v = self._reaction_step(u, v, half_dt)
            u, v = self._hamiltonian_step(u, v, gamma, dt)
            u, v = self._reaction_step(u, v, half_dt)

        # (B, D, H, W) → (B, D, N) → (B, N, D)  — NO reshape / view
        u = u.flatten(2).permute(0, 2, 1)                             # (B, N, D)
        v = v.flatten(2).permute(0, 2, 1)

        combined = torch.cat([u, v], dim=-1)                          # (B, N, 2D)
        return self.out_mix(combined) + x * self.D


# ---------------------------------------------------------------------------
# Outer Mamba Block (wraps the PDE operator)
# ---------------------------------------------------------------------------


class ContinuousSpatialMambaBlock_V5(nn.Module):
    """Drop-in block using the real-valued symplectic spatial operator."""

    def __init__(self, d_model: int, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        d_inner = int(expand * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.local_conv2d = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True,
        )
        self.activation = nn.SiLU()
        self.ssm = RealSymplecticReactionDiffusion2D(d_model=d_model, expand=expand)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, k_steps: int = 4) -> torch.Tensor:
        residual = x
        bsz, num_tokens, _ = x.shape
        h = w = int(math.sqrt(num_tokens))

        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)

        # (B, N, D) → (B, D, H, W) — no reshape, no view
        u_2d = u.permute(0, 2, 1).unflatten(2, (h, w))
        u_2d = self.local_conv2d(u_2d)
        # (B, D, H, W) → (B, N, D)
        u = u_2d.flatten(2).permute(0, 2, 1)
        u = self.activation(u)

        y_ssm = self.ssm(u, k_steps=k_steps)
        out = self.out_proj(y_ssm * F.silu(z))
        return residual + self.drop_path(out)


# ---------------------------------------------------------------------------
# Top-level Model
# ---------------------------------------------------------------------------


class CSMamba_V5(nn.Module):
    """
    CS-Mamba V5
    -----------
    Purely real-valued, TPU-native, symplectic reaction-diffusion backbone.
    """

    def __init__(self, cfg):
        super().__init__()
        img_size = getattr(cfg, "img_size", getattr(cfg, "canvas_size", 224))
        patch_size = getattr(cfg, "patch_size", 16)
        d_embed = getattr(cfg, "d_embed", 384)
        n_layers = getattr(cfg, "n_mamba_layers", 12)
        n_classes = getattr(cfg, "n_classes", 1000)
        k_steps = getattr(cfg, "K_steps", 4)
        drop_path_rate = getattr(cfg, "drop_path", 0.1)

        self.k_steps = k_steps
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([
            ContinuousSpatialMambaBlock_V5(d_model=d_embed, drop_path=dp_rates[i])
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CSMamba_V5 (Real Symplectic) %.1fM params, d=%d, layers=%d, K=%d",
            n_params / 1e6, d_embed, n_layers, k_steps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            x = layer(x, k_steps=self.k_steps)
        return self.head(self.final_norm(x).mean(dim=1))


__all__ = [
    "RealSymplecticReactionDiffusion2D",
    "ContinuousSpatialMambaBlock_V5",
    "CSMamba_V5",
]
