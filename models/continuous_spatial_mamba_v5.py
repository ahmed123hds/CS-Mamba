import math
import logging
from typing import Optional

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
        random_tensor = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * random_tensor / keep_prob


class RealSymplecticReactionDiffusion2D(nn.Module):
    """
    TPU-native real-valued Schrödinger-inspired block.

    State representation:
        psi = u + i v  is represented by two real tensors (u, v).

    Linear Hamiltonian flow:
        u_t =  gamma * Lap(v)
        v_t = -gamma * Lap(u)

    Nonlinear reaction flow:
        psi_t = psi * (alpha - beta |psi|^2)

    Discretization:
        Strang splitting with an exact scalar reaction step and a leapfrog
        (Störmer-Verlet style) symplectic update for the linear Hamiltonian step.

    This avoids backend-fragile complex FFT dependencies, so it is compatible with
    BF16 TPU execution and PyTorch XLA lowering.
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        # Sample-adaptive but spatially uniform diffusion strength per channel.
        self.diffusion_gate = nn.Linear(d_inner, d_inner, bias=True)
        nn.init.uniform_(self.diffusion_gate.weight, -1e-4, 1e-4)
        nn.init.constant_(self.diffusion_gate.bias, math.log(math.exp(0.08) - 1.0))
        self.log_diffusion_base = nn.Parameter(torch.full((1, d_inner), -2.5))
        self.max_diffusion = 0.25

        # Reaction parameters. Alpha is bounded; beta stays positive.
        self.reaction_alpha_raw = nn.Parameter(torch.zeros(1, 1, d_inner))
        self.reaction_beta_raw = nn.Parameter(torch.full((1, 1, d_inner), -2.0))
        self.alpha_scale = 0.20
        self.beta_eps = 1e-4

        # Final real-valued complex mixer + residual diagonal.
        self.out_complex = nn.Linear(d_inner * 2, d_inner, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

    @staticmethod
    def _laplacian_periodic(x: torch.Tensor) -> torch.Tensor:
        """Periodic 5-point Laplacian using only TPU-native real ops."""
        return (
            torch.roll(x, shifts=1, dims=-2)
            + torch.roll(x, shifts=-1, dims=-2)
            + torch.roll(x, shifts=1, dims=-1)
            + torch.roll(x, shifts=-1, dims=-1)
            - 4.0 * x
        )

    def _reaction_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = x.mean(dim=1)
        gamma = F.softplus(self.diffusion_gate(pooled) + self.log_diffusion_base)
        gamma = gamma.clamp(max=self.max_diffusion).unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)

        alpha = self.alpha_scale * torch.tanh(self.reaction_alpha_raw)
        beta = F.softplus(self.reaction_beta_raw) + self.beta_eps
        alpha = alpha.permute(0, 2, 1).unsqueeze(-1)  # (1, D, 1, 1)
        beta = beta.permute(0, 2, 1).unsqueeze(-1)    # (1, D, 1, 1)
        return gamma, alpha, beta

    @staticmethod
    def _exact_reaction_step(
        u: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exact amplitude update for q = u^2 + v^2 under
            dq/dt = 2 alpha q - 2 beta q^2.
        The phase is unchanged, so both u and v are scaled by the same factor.
        """
        eps = 1e-8
        q = u.square() + v.square()
        a = 2.0 * alpha
        b = 2.0 * beta

        exp_term = torch.exp(-a * dt)
        denom = b * q + (a - b * q) * exp_term
        q_general = (a * q) / denom.clamp_min(eps)
        q_zero_alpha = q / (1.0 + b * q * dt)
        q_new = torch.where(a.abs() < 1e-6, q_zero_alpha, q_general).clamp_min(0.0)

        scale = torch.sqrt(q_new / q.clamp_min(eps))
        return u * scale, v * scale

    def _leapfrog_hamiltonian_step(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        gamma: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Real-valued symplectic update for
            u_t =  gamma * Lap(v)
            v_t = -gamma * Lap(u)
        using a leapfrog/Störmer-Verlet style step.
        """
        v_half = v - 0.5 * dt * gamma * self._laplacian_periodic(u)
        u_new = u + dt * gamma * self._laplacian_periodic(v_half)
        v_new = v_half - 0.5 * dt * gamma * self._laplacian_periodic(u_new)
        return u_new, v_new

    def forward(self, x: torch.Tensor, k_steps: int = 4) -> torch.Tensor:
        bsz, num_tokens, d_inner = x.shape
        h = w = int(math.sqrt(num_tokens))
        if h * w != num_tokens:
            raise ValueError(f"Expected square token grid, got N={num_tokens}")

        gamma, alpha, beta = self._reaction_params(x)
        dt = 1.0 / max(int(k_steps), 1)

        # Keep dtype native to the incoming activation so BF16 on TPU stays fast.
        state_dtype = x.dtype
        u = x.transpose(1, 2).reshape(bsz, d_inner, h, w).to(state_dtype)
        v = torch.zeros_like(u)

        gamma = gamma.to(state_dtype)
        alpha = alpha.to(state_dtype)
        beta = beta.to(state_dtype)

        half_dt = 0.5 * dt
        for _ in range(int(k_steps)):
            u, v = self._exact_reaction_step(u, v, alpha, beta, half_dt)
            u, v = self._leapfrog_hamiltonian_step(u, v, gamma, dt)
            u, v = self._exact_reaction_step(u, v, alpha, beta, half_dt)

        u = u.reshape(bsz, d_inner, num_tokens).transpose(1, 2)
        v = v.reshape(bsz, d_inner, num_tokens).transpose(1, 2)
        combined = torch.cat([u, v], dim=-1)
        return self.out_complex(combined) + x * self.D.to(x.dtype)


class ContinuousSpatialMambaBlock_V5(nn.Module):
    """Drop-in block using the real-valued symplectic spatial operator."""

    def __init__(self, d_model: int, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        d_inner = int(expand * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.local_conv2d = nn.Conv2d(
            d_inner,
            d_inner,
            kernel_size=3,
            padding=1,
            groups=d_inner,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.ssm = RealSymplecticReactionDiffusion2D(d_model=d_model, expand=expand)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, k_steps: int = 4) -> torch.Tensor:
        residual = x
        bsz, num_tokens, _ = x.shape
        h = w = int(math.sqrt(num_tokens))
        if h * w != num_tokens:
            raise ValueError(f"Expected square token grid, got N={num_tokens}")

        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)

        u_2d = u.transpose(1, 2).reshape(bsz, -1, h, w)
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.reshape(bsz, -1, num_tokens).transpose(1, 2)
        u = self.activation(u)

        y_ssm = self.ssm(u, k_steps=k_steps)
        out = self.out_proj(y_ssm * F.silu(z))
        return residual + self.drop_path(out)


class CSMamba_V5(nn.Module):
    """
    CS-Mamba V5
    ------------
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
            n_params / 1e6,
            d_embed,
            n_layers,
            k_steps,
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
