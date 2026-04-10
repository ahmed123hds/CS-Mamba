"""
Continuous Spatial Mamba v3 — Learned Reaction-Diffusion
========================================================
Key change over v2:
  1. LEARNED Diffusion: Fixed Laplacian replaced with learned 3×3 DWConv
     per state dimension. The model learns WHICH directions to diffuse.
  2. Reaction Term: Gated MLP acts as a nonlinear "reaction" that CREATES
     discriminative spatial patterns instead of just blurring them.
  3. DropPath: Stochastic depth for better regularization.

Physics: The PDE changes from pure heat equation to learned reaction-diffusion:
  ∂h/∂t = Force_1(SSM) + Force_2(Learned Diffusion) + Force_3(Reaction)

This creates a Turing-like pattern formation system that generates
structured features rather than smoothing them away.
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


# ── Learned Reaction-Diffusion PDE Forward ──────────────────────
def cs_mamba_forward_v3(h0, x, delta_s, delta_d, A, B_mat, 
                         diffusion_conv, reaction_gate, reaction_proj,
                         K, H, W, mamba_input=None):
    """
    Learned Reaction-Diffusion PDE Integration.
    
    Instead of a fixed Laplacian (which blurs), this uses:
      - Learned DWConv for anisotropic, edge-aware diffusion
      - Gated MLP for nonlinear reaction (pattern creation)
    
    Args:
        h0:              (B, N, D, S) initial state
        x:               (B, N, D) input features
        delta_s:         (B, N, D) self-decay time scale
        delta_d:         (B, N, D) diffusion time scale
        A:               (D, S) state decay matrix
        B_mat:           (B, N, S) input projection
        diffusion_conv:  nn.Conv2d — learned spatial mixing operator
        reaction_gate:   nn.Linear — gate for reaction term
        reaction_proj:   nn.Linear — projection for reaction term
        K:               int, number of Euler steps
        H, W:            spatial grid dimensions
    """
    B_val, N, D_dim, S_dim = h0.shape
    dt = 1.0 / K
    h = h0.clone()

    if mamba_input is None:
        mamba_input = torch.einsum('bnd,bns->bnds', x, B_mat)

    A_exp = A.view(1, 1, D_dim, S_dim)

    for k in range(K):
        # ── Force 1: SSM Recurrence (same as V2) ─────────────────
        force_1 = delta_s.unsqueeze(-1) * (A_exp * h + mamba_input)

        # ── Force 2: LEARNED Spatial Diffusion ───────────────────
        # Reshape: (B, N, D, S) → permute to (B, S, D, N) → (B*S, D, H, W)
        h_spatial = h.permute(0, 3, 2, 1).reshape(B_val * S_dim, D_dim, H, W)
        
        # Learned DWConv replaces fixed Laplacian
        # The kernel learns direction-dependent, edge-aware diffusion
        h_pad = F.pad(h_spatial, (1, 1, 1, 1), mode='replicate')
        diff_h_2d = diffusion_conv(h_pad)
        
        # Reshape back: (B*S, D, H, W) → (B, S, D, N) → (B, N, D, S)
        diff_h = diff_h_2d.reshape(B_val, S_dim, D_dim, N).permute(0, 3, 2, 1)
        force_2 = delta_d.unsqueeze(-1) * diff_h

        # ── Force 3: Nonlinear Reaction ──────────────────────────
        # Gated MLP creates discriminative patterns instead of just blurring
        # Acts pointwise on each spatial location
        h_flat = h.reshape(B_val * N, D_dim * S_dim)
        gate = torch.sigmoid(reaction_gate(h_flat))
        reaction = gate * reaction_proj(h_flat)
        force_3 = reaction.reshape(B_val, N, D_dim, S_dim)

        # ── Explicit Euler Step ──────────────────────────────────
        h = h + dt * (force_1 + force_2 + force_3)

    return h


class ContinuousSpatialSSM_V3(nn.Module):
    """
    Learned Reaction-Diffusion Spatial SSM
    =======================================
    Key innovations:
      - Learned 3×3 DWConv replaces fixed Laplacian (anisotropic diffusion)
      - Gated MLP reaction term (Turing pattern formation)
      - CFL stability maintained via softplus + clamp
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(expand * d_model)

        # Dual Input-dependent Time-Scales
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
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

        # S4-style log-decay
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(d_inner, -1)
        )

        # ── LEARNED Diffusion Operator (replaces fixed Laplacian) ──
        # 3×3 depthwise conv with learnable kernel per feature channel
        # Initialized to approximate Laplacian but FREE to learn any pattern
        self.diffusion_conv = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=0,
            groups=d_inner, bias=False
        )
        # Initialize weights to approximate Laplacian (good starting point)
        with torch.no_grad():
            lap_init = torch.tensor(
                [[0.0, 1.0, 0.0],
                 [1.0, -4.0, 1.0],
                 [0.0, 1.0, 0.0]], dtype=torch.float32
            ).view(1, 1, 3, 3) * 0.1  # Scale down for stability
            self.diffusion_conv.weight.copy_(lap_init.repeat(d_inner, 1, 1, 1))

        # ── Reaction Term (Gated MLP) ──
        # Projects from (D*S) → (D*S) with a sigmoid gate
        reaction_dim = d_inner * d_state
        self.reaction_gate = nn.Linear(reaction_dim, reaction_dim, bias=True)
        self.reaction_proj = nn.Linear(reaction_dim, reaction_dim, bias=False)
        # Initialize reaction near zero so model starts close to V2 behavior
        nn.init.zeros_(self.reaction_gate.weight)
        nn.init.zeros_(self.reaction_gate.bias)
        nn.init.uniform_(self.reaction_proj.weight, -1e-4, 1e-4)

    def forward(self, x: torch.Tensor, K_steps: int = 2, use_triton: bool = False) -> torch.Tensor:
        B_val, N, D_dim = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Spatial Mamba requires N to be a perfect square."

        A_mat = -F.softplus(self.A_log)

        delta_self = torch.clamp(F.softplus(self.dt_self_proj(x)), max=0.15)
        delta_diff = torch.clamp(F.softplus(self.dt_diff_proj(x)), max=0.15)

        B_mat = self.B_proj(x)
        C_mat = self.C_proj(x)

        h0 = torch.einsum('bnd,bns->bnds', x, B_mat)

        h = cs_mamba_forward_v3(
            h0, x, delta_self, delta_diff, A_mat, B_mat,
            self.diffusion_conv, self.reaction_gate, self.reaction_proj,
            K_steps, H, W, mamba_input=h0
        )

        y = torch.einsum('bnds,bns->bnd', h, C_mat)
        y = y + x * self.D
        return y


class ContinuousSpatialMambaBlock_V3(nn.Module):
    """
    Reaction-Diffusion Mamba Block with DropPath.
    Drop-in replacement for V2 block.
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
        self.continuous_ssm = ContinuousSpatialSSM_V3(
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

        # PDE Integration (Reaction-Diffusion)
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


class CSMamba_V3(nn.Module):
    """
    CS-Mamba V3 — Learned Reaction-Diffusion Architecture
    ======================================================
    Uses learned anisotropic diffusion + nonlinear reaction for
    pattern-creating spatial feature mixing. Replaces information-
    destroying heat equation with pattern-generating reaction-diffusion.
    """
    def __init__(self, cfg):
        super().__init__()
        img_size = getattr(cfg, 'img_size', getattr(cfg, 'canvas_size', 224))
        patch_size = getattr(cfg, 'patch_size', 16)
        d_embed = getattr(cfg, 'd_embed', 384)
        n_layers = getattr(cfg, 'n_mamba_layers', 12)
        d_state = getattr(cfg, 'd_state', 16)
        n_classes = getattr(cfg, 'n_classes', 1000)
        K_steps = getattr(cfg, 'K_steps', 2)  # Default K=2 (less blurring)
        drop_path_rate = getattr(cfg, 'drop_path', 0.1)

        self.K_steps = K_steps
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stochastic depth schedule (linearly increasing)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        self.layers = nn.ModuleList([
            ContinuousSpatialMambaBlock_V3(
                d_model=d_embed, d_state=d_state,
                drop_path=dp_rates[i]
            )
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"CSMamba_V3 initialized: {n_params/1e6:.1f}M params, "
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
