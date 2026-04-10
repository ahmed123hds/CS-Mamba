"""
Continuous Spatial Mamba v4 — Complex Schrödinger Reaction-Diffusion
====================================================================
Key insight: Replace the REAL heat equation (which blurs/destroys features)
with the COMPLEX Schrödinger equation (which preserves information via
unitary evolution and creates interference patterns).

Physics:
  Heat Eq  (V2):  ∂h/∂t = D∇²h          → dissipative, blurs features
  Schrödinger (V4): ∂ψ/∂t = iD∇²ψ + R(ψ) → unitary + reaction patterns

Implementation:
  - ψ = h_real + i·h_imag  (two real tensors, XLA-compatible)
  - i * laplacian(ψ) = [-laplacian(h_imag), +laplacian(h_real)]
  - Real and imaginary parts cross-couple through spatial diffusion
  - Reaction term operates on magnitude |ψ| for nonlinear pattern creation
  - Output: learned projection of both real and imaginary components

No complex dtypes needed — all arithmetic is real-valued for TPU/XLA.

References:
  - S4 (Gu et al.): Complex diagonal SSM → breakthrough on long-range tasks
  - S4D: Complex eigenvalues dramatically improve SSM representations
  - CKConv (Romero et al.): Complex-valued continuous kernels for vision
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Stochastic Depth — drops entire residual branches during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ── Complex Schrödinger PDE Forward ─────────────────────────
def cs_mamba_forward_v4(h0_real, h0_imag, x, delta_s, delta_d, A, B_mat,
                         diffusion_conv, reaction_alpha, reaction_beta,
                         K, H, W, mamba_input=None):
    """
    Complex Schrödinger Reaction-Diffusion PDE Integration.
    
    The hidden state is complex: ψ = h_real + i·h_imag
    Diffusion uses Schrödinger cross-coupling (unitary, norm-preserving):
      ∂ψ/∂t = i·D·∇²ψ  →  ∂h_real/∂t = -D·∇²h_imag
                             ∂h_imag/∂t = +D·∇²h_real
    
    Args:
        h0_real, h0_imag: (B, N, D, S) initial state (real & imaginary)
        x:                (B, N, D) input features
        delta_s:          (B, N, D) SSM time scale
        delta_d:          (B, N, D) diffusion time scale
        A:                (D, S) state decay matrix
        B_mat:            (B, N, S) input projection
        diffusion_conv:   nn.Conv2d — learned spatial operator
        reaction_alpha:   (1, 1, D, 1) reaction alpha coefficient
        reaction_beta:    (1, 1, D, 1) reaction beta coefficient
        K:                int, Euler steps
        H, W:             spatial grid dimensions
    """
    B_val, N, D_dim, S_dim = h0_real.shape
    dt = 1.0 / K
    h_real = h0_real.clone()
    h_imag = h0_imag.clone()

    if mamba_input is None:
        mamba_input = torch.einsum('bnd,bns->bnds', x, B_mat)

    A_exp = A.view(1, 1, D_dim, S_dim)

    for k in range(K):
        # ── Force 1: SSM Recurrence (on both real & imag) ────────
        # The SSM decay operates independently on each component
        f1_real = delta_s.unsqueeze(-1) * (A_exp * h_real + mamba_input)
        f1_imag = delta_s.unsqueeze(-1) * (A_exp * h_imag)  # No input to imag

        # ── Force 2: Schrödinger Cross-Coupled Diffusion ─────────
        # i * D * laplacian(ψ) = [-D·lap(h_imag), +D·lap(h_real)]
        # This is the KEY: real and imaginary parts CROSS-COUPLE
        # through the spatial operator, creating interference patterns.

        # Apply learned spatial operator to REAL part
        h_r_spatial = h_real.permute(0, 3, 2, 1).reshape(B_val * S_dim, D_dim, H, W)
        h_r_pad = F.pad(h_r_spatial, (1, 1, 1, 1), mode='replicate')
        lap_real = diffusion_conv(h_r_pad)
        lap_real = lap_real.reshape(B_val, S_dim, D_dim, N).permute(0, 3, 2, 1)

        # Apply learned spatial operator to IMAGINARY part
        h_i_spatial = h_imag.permute(0, 3, 2, 1).reshape(B_val * S_dim, D_dim, H, W)
        h_i_pad = F.pad(h_i_spatial, (1, 1, 1, 1), mode='replicate')
        lap_imag = diffusion_conv(h_i_pad)
        lap_imag = lap_imag.reshape(B_val, S_dim, D_dim, N).permute(0, 3, 2, 1)

        # Schrödinger cross-coupling:  i * lap(ψ) = [-lap(imag), +lap(real)]
        f2_real = delta_d.unsqueeze(-1) * (-lap_imag)  # ← imag feeds real
        f2_imag = delta_d.unsqueeze(-1) * (+lap_real)  # ← real feeds imag

        # ── Force 3: Nonlinear Reaction (Ginzburg-Landau) ─────────────
        # Ginzburg-Landau equation for complex scalar fields: R(ψ) = ψ·(α - β|ψ|²)
        # This completely avoids sqrt() and division, preventing XLA compiler OOM!
        mag_sq = h_real ** 2 + h_imag ** 2
        reaction_scale = reaction_alpha - reaction_beta * mag_sq

        f3_real = h_real * reaction_scale
        f3_imag = h_imag * reaction_scale

        # ── Explicit Euler Step ──────────────────────────────────
        h_real = h_real + dt * (f1_real + f2_real + f3_real)
        h_imag = h_imag + dt * (f1_imag + f2_imag + f3_imag)

    return h_real, h_imag


class ContinuousSpatialSSM_V4(nn.Module):
    """
    Complex Schrödinger Spatial SSM
    ================================
    Key innovations over V3:
      - Complex hidden state ψ = h_real + i·h_imag
      - Schrödinger cross-coupling: unitary, norm-preserving diffusion
      - Reaction on magnitude |ψ| for phase-aware pattern creation
      - Output: learned linear combination of real & imaginary features
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(expand * d_model)

        # Dual Time-Scales
        self.dt_self_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.dt_diff_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Stability init
        dt_init = math.log(math.exp(0.1) - 1.0)
        nn.init.constant_(self.dt_self_proj.bias, dt_init)
        nn.init.constant_(self.dt_diff_proj.bias, dt_init)
        nn.init.uniform_(self.dt_self_proj.weight, -1e-4, 1e-4)
        nn.init.uniform_(self.dt_diff_proj.weight, -1e-4, 1e-4)

        # Mamba Projections
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

        # S4-style log-decay
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(d_inner, -1)
        )

        # Output: DUAL C projections for real and imaginary readout
        # y = C_real @ h_real + C_imag @ h_imag  (learned combination)
        self.C_real_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_imag_proj = nn.Linear(d_inner, d_state, bias=False)

        # ── Learned Spatial Operator (shared for both components) ──
        self.diffusion_conv = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=0,
            groups=d_inner, bias=False
        )
        # Initialize to approximate Laplacian
        with torch.no_grad():
            lap_init = torch.tensor(
                [[0.0, 1.0, 0.0],
                 [1.0, -4.0, 1.0],
                 [0.0, 1.0, 0.0]], dtype=torch.float32
            ).view(1, 1, 3, 3) * 0.1
            self.diffusion_conv.weight.copy_(lap_init.repeat(d_inner, 1, 1, 1))

        # ── Reaction Term (Ginzburg-Landau on |ψ|²) ──
        # Phase separation reaction: ψ·(α - β|ψ|²)
        self.reaction_alpha = nn.Parameter(torch.zeros(1, 1, d_inner, 1))
        self.reaction_beta = nn.Parameter(torch.zeros(1, 1, d_inner, 1))

    def forward(self, x: torch.Tensor, K_steps: int = 2, use_triton: bool = False) -> torch.Tensor:
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Spatial Mamba requires N to be a perfect square."

        A_mat = -F.softplus(self.A_log)

        delta_self = torch.clamp(F.softplus(self.dt_self_proj(x)), max=0.15)
        delta_diff = torch.clamp(F.softplus(self.dt_diff_proj(x)), max=0.15)

        B_mat = self.B_proj(x)
        C_real = self.C_real_proj(x)  # (B, N, S)
        C_imag = self.C_imag_proj(x)  # (B, N, S)

        # Initialize: real part from input, imaginary part = 0
        # The Schrödinger evolution will generate the imaginary component
        h0_real = torch.einsum('bnd,bns->bnds', x, B_mat)
        h0_imag = torch.zeros_like(h0_real)

        # Complex PDE Integration
        h_real, h_imag = cs_mamba_forward_v4(
            h0_real, h0_imag, x, delta_self, delta_diff, A_mat, B_mat,
            self.diffusion_conv, self.reaction_alpha, self.reaction_beta,
            K_steps, H, W, mamba_input=h0_real
        )

        # Output: learned combination of both complex components
        # y = C_real·h_real + C_imag·h_imag + D·x
        y_real = torch.einsum('bnds,bns->bnd', h_real, C_real)
        y_imag = torch.einsum('bnds,bns->bnd', h_imag, C_imag)
        y = y_real + y_imag + x * self.D

        return y


class ContinuousSpatialMambaBlock_V4(nn.Module):
    """
    Complex Schrödinger Mamba Block.
    Drop-in replacement — same interface as V2/V3 blocks.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_model, expand * d_model * 2, bias=False)

        # Native 2D Pre-Processing
        self.local_conv2d = nn.Conv2d(
            in_channels=expand * d_model, out_channels=expand * d_model,
            kernel_size=3, padding=1, groups=expand * d_model, bias=True
        )

        self.activation = nn.SiLU()
        self.continuous_ssm = ContinuousSpatialSSM_V4(
            d_model=d_model, d_state=d_state, expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(expand * d_model, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, K_steps: int = 2, use_triton: bool = False) -> torch.Tensor:
        residual = x
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))

        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)

        # 2D Local Convolution
        u_2d = u.transpose(1, 2).view(B_val, -1, H, W)
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.view(B_val, -1, N).transpose(1, 2)
        u = self.activation(u)

        # Complex PDE Integration
        if u.device.type == "xla":
            y_ssm = self.continuous_ssm(u, K_steps=K_steps, use_triton=use_triton)
        else:
            from torch.utils.checkpoint import checkpoint
            y_ssm = checkpoint(
                self.continuous_ssm, u,
                K_steps, use_triton,
                use_reentrant=False,
            )

        # Gated output + DropPath
        out = self.out_proj(y_ssm * F.silu(z))
        return residual + self.drop_path(out)


class CSMamba_V4(nn.Module):
    """
    CS-Mamba V4 — Complex Schrödinger Reaction-Diffusion
    =====================================================
    The first complex-valued spatial SSM for vision.
    
    Instead of blurring features (heat equation), it creates
    constructive/destructive interference patterns (Schrödinger)
    that encode rich spatial relationships while preserving
    information (unitary evolution).
    
    The hidden state ψ = h_real + i·h_imag evolves via:
      ∂ψ/∂t = F_SSM(ψ) + i·D·∇²ψ + R(|ψ|)·ψ/|ψ|
    """
    def __init__(self, cfg):
        super().__init__()
        img_size = getattr(cfg, 'img_size', getattr(cfg, 'canvas_size', 224))
        patch_size = getattr(cfg, 'patch_size', 16)
        d_embed = getattr(cfg, 'd_embed', 384)
        n_layers = getattr(cfg, 'n_mamba_layers', 12)
        d_state = getattr(cfg, 'd_state', 16)
        n_classes = getattr(cfg, 'n_classes', 1000)
        K_steps = getattr(cfg, 'K_steps', 2)
        drop_path_rate = getattr(cfg, 'drop_path', 0.1)

        self.K_steps = K_steps
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stochastic depth schedule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        self.layers = nn.ModuleList([
            ContinuousSpatialMambaBlock_V4(
                d_model=d_embed, d_state=d_state,
                drop_path=dp_rates[i]
            )
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"CSMamba_V4 (Complex Schrödinger) initialized: {n_params/1e6:.1f}M params, "
                    f"d={d_embed}, layers={n_layers}, K={K_steps}, "
                    f"drop_path={drop_path_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x, K_steps=self.K_steps)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x)
