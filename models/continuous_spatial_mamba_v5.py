import math
import logging
from typing import Dict, Tuple

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
        random_tensor = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        )
        return x * random_tensor / keep_prob


class ExactReactionDiffusion2D(nn.Module):
    """
    Structure-preserving 2D Schrödinger + Ginzburg-Landau block.

    Key design changes relative to V4:
      1) The diffusion step is *exactly* integrated in the spectral domain.
      2) The spatial operator is a fixed periodic 2D Laplacian, so it is self-adjoint.
      3) The reaction step is integrated in closed form on |psi|^2.
      4) We use Strang splitting: half reaction -> full diffusion -> half reaction.

    This makes the discrete diffusion step genuinely norm-preserving instead of
    merely inspired by unitary continuous-time dynamics.
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        # Sample-adaptive but spatially uniform diffusion coefficient per channel.
        # This keeps the discrete spatial operator self-adjoint for every sample.
        self.diffusion_gate = nn.Linear(d_inner, d_inner, bias=True)
        nn.init.uniform_(self.diffusion_gate.weight, -1e-4, 1e-4)
        nn.init.constant_(self.diffusion_gate.bias, math.log(math.exp(0.08) - 1.0))
        self.log_diffusion_base = nn.Parameter(torch.full((1, d_inner), -2.5))
        self.max_diffusion = 0.35

        # Bounded reaction parameters for stable amplitude dynamics.
        self.reaction_alpha_raw = nn.Parameter(torch.zeros(1, 1, d_inner))
        self.reaction_beta_raw = nn.Parameter(torch.full((1, 1, d_inner), -2.0))
        self.alpha_scale = 0.25
        self.beta_eps = 1e-4

        # Complex output mixer + skip scale.
        self.out_complex = nn.Linear(d_inner * 2, d_inner, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

        self._laplacian_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _laplacian_eigs(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device))
        cached = self._laplacian_cache.get(key)
        if cached is not None:
            return cached

        ky = torch.arange(h, device=device, dtype=torch.float32)
        kx = torch.arange(w, device=device, dtype=torch.float32)
        eig_y = 2.0 * torch.cos(2.0 * math.pi * ky / h) - 2.0
        eig_x = 2.0 * torch.cos(2.0 * math.pi * kx / w) - 2.0
        eig = eig_y[:, None] + eig_x[None, :]  # values in [-8, 0]
        self._laplacian_cache[key] = eig
        return eig

    def _exact_reaction_step(self, psi: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Exact update for q = |psi|^2 under
            dq/dt = 2 alpha q - 2 beta q^2.
        Phase stays unchanged; only amplitude changes.
        """
        eps = 1e-8
        q = psi.real.square() + psi.imag.square()
        a = 2.0 * alpha
        b = 2.0 * beta

        small_mask = a.abs() < 1e-6
        exp_term = torch.exp(-a * dt)
        denom = b * q + (a - b * q) * exp_term
        q_general = (a * q) / denom.clamp_min(eps)
        q_zero_alpha = q / (1.0 + b * q * dt)
        q_new = torch.where(small_mask, q_zero_alpha, q_general).clamp_min(0.0)

        scale = torch.sqrt(q_new / q.clamp_min(eps))
        return psi * scale.to(psi.dtype)

    def _exact_diffusion_step(self, psi: torch.Tensor, gamma: torch.Tensor, lap_eigs: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Exact discrete Schrödinger flow for
            d psi / dt = i * gamma * Laplacian(psi)
        with periodic boundary conditions.
        """
        psi_hat = torch.fft.fft2(psi, dim=(-2, -1))
        phase_arg = dt * gamma[:, :, None, None] * lap_eigs[None, None, :, :]
        phase = torch.complex(torch.cos(phase_arg), torch.sin(phase_arg))
        psi_hat = psi_hat * phase
        return torch.fft.ifft2(psi_hat, dim=(-2, -1))

    def forward(self, x: torch.Tensor, k_steps: int = 4) -> torch.Tensor:
        bsz, num_tokens, d_inner = x.shape
        h = w = int(math.sqrt(num_tokens))
        if h * w != num_tokens:
            raise ValueError(f"Expected square token grid, got N={num_tokens}")

        # Sample-adaptive but spatially uniform diffusion coefficient.
        pooled = x.mean(dim=1)
        gamma = F.softplus(self.diffusion_gate(pooled) + self.log_diffusion_base)
        gamma = gamma.clamp(max=self.max_diffusion).to(torch.float32)

        alpha = (self.alpha_scale * torch.tanh(self.reaction_alpha_raw)).permute(0, 2, 1).unsqueeze(-1)
        beta = (F.softplus(self.reaction_beta_raw) + self.beta_eps).permute(0, 2, 1).unsqueeze(-1)
        lap_eigs = self._laplacian_eigs(h, w, x.device)
        dt = 1.0 / max(int(k_steps), 1)

        # Work in fp32 inside the PDE block for numerical stability.
        xr = x.to(torch.float32)
        psi = torch.complex(
            xr.transpose(1, 2).reshape(bsz, d_inner, h, w),
            torch.zeros(bsz, d_inner, h, w, device=x.device, dtype=torch.float32),
        )

        for _ in range(int(k_steps)):
            psi = self._exact_reaction_step(psi, alpha, beta, 0.5 * dt)
            psi = self._exact_diffusion_step(psi, gamma, lap_eigs, dt)
            psi = self._exact_reaction_step(psi, alpha, beta, 0.5 * dt)

        h_real = psi.real.reshape(bsz, d_inner, num_tokens).transpose(1, 2)
        h_imag = psi.imag.reshape(bsz, d_inner, num_tokens).transpose(1, 2)
        combined = torch.cat([h_real, h_imag], dim=-1).to(x.dtype)
        return self.out_complex(combined) + x * self.D.to(x.dtype)


class ContinuousSpatialMambaBlock_V5(nn.Module):
    """
    Exact reaction-diffusion block dropped into the same outer scaffold.
    """

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
        self.ssm = ExactReactionDiffusion2D(d_model=d_model, expand=expand)
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
    The diffusion step is now exactly norm-preserving in the discrete model.
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
            "CSMamba_V5 (Exact Schrödinger Split-Step) %.1fM params, d=%d, layers=%d, K=%d",
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
