"""
Continuous Spatial Mamba v2 — State-Preserving Diffusion
========================================================
Fixes over v1:
  1. State-Preserving Diffusion: Laplacian applied per-state-dim (B*S, D, H, W)
     instead of collapsing S via sum.
  2. F.conv2d Laplacian: Uses fused depthwise conv2d instead of manual slicing.
  3. Pre-computation: mamba_input moved outside the K loop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ── Optimized PDE Forward (State-Preserving Diffusion) ─────────────
def cs_mamba_forward_v2(h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W, laplacian_kernel, mamba_input=None):
    """
    State-Preserving Continuous Spatial PDE Integration.
    
    Key fix: The Laplacian is applied to each of the S state dimensions
    independently, preserving the full expressiveness of the SSM latent space.
    
    Args:
        h0:               (B, N, D, S) initial state
        x:                (B, N, D) input features
        delta_s:          (B, N, D) self-decay time scale
        delta_d:          (B, N, D) diffusion time scale
        A:                (D, S)   state decay matrix (negative)
        B_mat:            (B, N, S) input projection
        D_phys:           (1, D, 1, 1) diffusivity coefficient
        K:                int, number of Euler steps
        H, W:             spatial grid dimensions
        laplacian_kernel: (D, 1, 3, 3) fixed Laplacian conv kernel
    """
    B_val, N, D_dim, S_dim = h0.shape
    dt = 1.0 / K
    h = h0.clone()

    # Pre-compute input forcing (does NOT depend on h, constant across K steps)
    if mamba_input is None:
        mamba_input = torch.einsum('bnd,bns->bnds', x, B_mat)  # (B, N, D, S)

    # Pre-expand constants
    A_exp = A.view(1, 1, D_dim, S_dim)         # (1, 1, D, S)
    D_coeff = D_phys.view(1, 1, D_dim, 1)      # (1, 1, D, 1)

    for k in range(K):
        # ── FIX: State-Preserving Diffusion ──────────────────────
        # Reshape so each state dimension gets its OWN spatial diffusion.
        # h: (B, N, D, S) → permute to (B, S, N, D) → reshape to (B*S, D, H, W)
        h_spatial = h.permute(0, 3, 2, 1).reshape(B_val * S_dim, D_dim, H, W)

        # Apply Neumann BC (replicate padding) + fixed Laplacian via F.conv2d
        h_pad = F.pad(h_spatial, (1, 1, 1, 1), mode='replicate')
        lap_h_2d = F.conv2d(h_pad, laplacian_kernel, groups=D_dim)

        # Reshape back: (B*S, D, H, W) → (B, S, D, N) → (B, N, D, S)
        lap_h = lap_h_2d.reshape(B_val, S_dim, D_dim, N).permute(0, 3, 2, 1)

        # ── Force 1: Internal Mamba Decay + Input ────────────────
        force_1 = delta_s.unsqueeze(-1) * (A_exp * h + mamba_input)

        # ── Force 2: Thermodynamic Spatial Diffusion ─────────────
        force_2 = delta_d.unsqueeze(-1) * D_coeff * lap_h

        # ── Explicit Euler Step ──────────────────────────────────
        h = h + dt * (force_1 + force_2)

    return h


class ContinuousSpatialSSM_V2(nn.Module):
    """
    Physics-Informed Continuous Spatial PDE Mamba V2
    ================================================
    Fixes:
      - Per-state diffusion (no S-dimension collapse)
      - F.conv2d Laplacian (fused depthwise, faster on GPU/TPU)
      - Pre-computed mamba_input outside K loop
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(expand * d_model)

        # Dual Input-dependent Time-Scales (Gates)
        self.dt_self_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.dt_diff_proj = nn.Linear(d_inner, d_inner, bias=True)

        # --- CRITICAL ODE STABILITY INITIALIZATION ---
        dt_init = math.log(math.exp(0.1) - 1.0)
        nn.init.constant_(self.dt_self_proj.bias, dt_init)
        nn.init.constant_(self.dt_diff_proj.bias, dt_init)
        nn.init.uniform_(self.dt_self_proj.weight, -1e-4, 1e-4)
        nn.init.uniform_(self.dt_diff_proj.weight, -1e-4, 1e-4)

        # Mamba Projections
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.D     = nn.Parameter(torch.ones(d_inner))

        # S4 / Mamba Log-decay initialization
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(d_inner, -1)
        )

        # --- PHYSICS-INFORMED THERMODYNAMIC DIFFUSION (F.conv2d version) ---
        # Fixed Laplacian ∇² kernel for depthwise conv2d
        lap_k = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", lap_k.repeat(d_inner, 1, 1, 1))

        # Learnable Diffusivity Constant (initialized to 0 for stability)
        self.diffusivity_raw = nn.Parameter(torch.zeros(1, d_inner, 1, 1))

    def forward(self, x: torch.Tensor, K_steps: int = 3, use_triton: bool = False) -> torch.Tensor:
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Spatial Mamba requires N to be a perfect square."

        # Enforce Re(lambda) < 0 for stability
        A_mat = -F.softplus(self.A_log)  # (D, S) < 0

        # Rigorous Euler Bound Clamp
        delta_self = torch.clamp(F.softplus(self.dt_self_proj(x)), max=0.15)  # (B, N, D)
        delta_diff = torch.clamp(F.softplus(self.dt_diff_proj(x)), max=0.15)  # (B, N, D)

        B_mat = self.B_proj(x)  # (B, N, S)
        C_mat = self.C_proj(x)  # (B, N, S)

        # Initialize PDE State: h(0) = B(x) ⊗ x
        h0 = torch.einsum('bnd,bns->bnds', x, B_mat)

        # Physics constraint: D MUST be > 0. Max bound 0.5 for CFL.
        D_phys = torch.sigmoid(self.diffusivity_raw) * 0.5

        # Explicit Euler Spatial Diffusion Loop (State-Preserving)
        h = cs_mamba_forward_v2(
            h0, x, delta_self, delta_diff, A_mat, B_mat,
            D_phys, K_steps, H, W, self.laplacian,
            mamba_input=h0 # Optimization: h0 and mamba_input are identical
        )

        # Output Projection: y(T) = C * h(T) + D * x
        y = torch.einsum('bnds,bns->bnd', h, C_mat)
        y = y + x * self.D
        return y


class ContinuousSpatialMambaBlock_V2(nn.Module):
    """
    Drop-in block natively avoiding 1D sequential bias entirely.
    Uses ContinuousSpatialSSM_V2 with state-preserving diffusion.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(d_model, expand * d_model * 2, bias=False)

        # Native 2D Pre-Processing (Not 1D Sequence Flat!)
        self.local_conv2d = nn.Conv2d(
            in_channels=expand * d_model, out_channels=expand * d_model,
            kernel_size=3, padding=1, groups=expand * d_model, bias=True
        )

        self.activation = nn.SiLU()
        self.continuous_ssm = ContinuousSpatialSSM_V2(
            d_model=d_model, d_state=d_state, expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(expand * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, K_steps: int = 3, use_triton: bool = False) -> torch.Tensor:
        residual = x
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))

        x_norm = self.norm(x)

        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)

        # ── 2D Local Convolution (No 1D flattening) ──
        u_2d = u.transpose(1, 2).view(B_val, -1, H, W)
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.view(B_val, -1, N).transpose(1, 2)

        u = self.activation(u)

        # ── PDE Integration (State-Preserving) ──
        # NOTE: gradient checkpointing removed from GPU path — it causes infinite
        # compilation hangs when combined with torch.compile.
        # Use AMP (autocast fp16) for memory savings on GPU instead.
        y_ssm = self.continuous_ssm(u, K_steps=K_steps, use_triton=use_triton)

        y = y_ssm * F.silu(z)
        y = self.out_proj(y)

        return y + residual


class CSMamba_V2(nn.Module):
    """Top-level classifier using CS-Mamba V2 blocks."""
    def __init__(self, cfg):
        super().__init__()
        from models.patch_encoder import PatchEmbedding
        self.embedder = PatchEmbedding(
            img_size=getattr(cfg, 'canvas_size', 128),
            patch_size=cfg.patch_size,
            in_channels=3,
            d_embed=cfg.d_embed,
        )
        self.blocks = nn.ModuleList([
            ContinuousSpatialMambaBlock_V2(cfg.d_embed, cfg.d_state)
            for _ in range(cfg.n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_embed)
        self.head = nn.Linear(cfg.d_embed, getattr(cfg, 'n_classes', 10))
        self.K_steps = getattr(cfg, 'K_steps', 3)

    def forward(self, x, use_triton: bool = False):
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x, K_steps=self.K_steps, use_triton=use_triton)
        x = self.norm(x)
        features = x.mean(dim=1)
        return self.head(features)
