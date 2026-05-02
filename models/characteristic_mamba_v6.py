"""
Characteristic Mamba V6 — Selective 2D Characteristic Transport
================================================================
Key ideas (from char_mamba_methodology.pdf):
  1. Learn a per-location transport direction via a stream function ψ
     whose curl gives a divergence-free velocity field.
  2. Convert state-derived velocity + learned self score to 9-direction routing weights.
  3. Transport paired state (U, V) by directional projected neighbor gathering.
  4. Optionally rotate the (U, V) pair by a learned phase angle Θ
     to retain V4's oscillatory channel coupling intuition.
  5. Update state via selective continuous-time retain/inject gating.

All ops are real-valued, use only standard conv/gather/softmax,
and are fully compatible with TPU/XLA BF16 execution.
"""

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
# Characteristic Transport Core
# ---------------------------------------------------------------------------

class CharacteristicTransport2D(nn.Module):
    """
    Selective 2D characteristic transport on the image plane.

    For T substeps, this module:
      1. Predicts a stream function ψ from the current state → derives velocity
      2. Converts velocity + learned self score to 9-direction routing weights
      3. Transports paired state (U, V) via directional projected neighbor gather
      4. Applies optional phase rotation on (U, V) pairs
      5. Updates state via continuous-time decay-derived retain/inject gates

    All operations use real-valued tensors with standard ops (F.conv2d,
    F.pad, softmax) — no FFT, no complex dtype, fully XLA/TPU safe.
    """

    def __init__(self, d_model: int, expand: int = 2, n_flow_groups: int = 4,
                 proj_drop: float = 0.0):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.n_flow_groups = n_flow_groups
        d_per_group = d_inner // n_flow_groups

        # --- Stream function head (predicts scalar ψ per flow group) ---
        # ψ is predicted per-group; velocity = curl(ψ) via finite differences
        self.stream_head = nn.Sequential(
            nn.Linear(d_inner, d_inner, bias=True),
            nn.SiLU(),
            nn.Linear(d_inner, n_flow_groups, bias=True),
        )
        self.self_route_head = nn.Linear(d_inner, n_flow_groups, bias=True)
        nn.init.zeros_(self.self_route_head.weight)
        nn.init.constant_(self.self_route_head.bias, 1.0)

        # Direction-specific value scaling for self + 8 neighbors.
        self.n_transport_dirs = 9
        self.direction_value_scale = nn.Parameter(
            torch.ones(1, d_inner, self.n_transport_dirs, 1, 1)
        )

        # --- Phase rotation head (optional, per-channel angle Θ) ---
        self.phase_head = nn.Linear(d_inner, d_inner, bias=True)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)

        # --- Selective continuous-time decay gate (NEW) ---
        self.delta_head = nn.Linear(d_inner, d_inner, bias=True)
        # self.routing_state_mix = nn.Parameter(torch.zeros(d_inner))

        # --- Old Gating (Sigmoid) ---
        # [ABLATION] Commented out old sigmoid gating
        # self.gate_a_head = nn.Linear(d_inner, d_inner, bias=True)
        # self.gate_b_head = nn.Linear(d_inner, d_inner, bias=True)

        # --- Injection features ---
        self.inj_proj = nn.Linear(d_inner, d_inner * 2, bias=False)

        # --- State initialization (project input to paired state) ---
        self.state_proj = nn.Linear(d_inner, d_inner * 2, bias=False)

        # --- Output: project (U, V) back to d_inner + skip ---
        self.out_proj = nn.Linear(d_inner * 2, d_inner, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.proj_drop = nn.Dropout(proj_drop)

        # Init gate to be mildly retentive at start:
        # softplus(-1.5) ≈ 0.20, exp(-0.20) ≈ 0.82 retain / 0.18 inject.
        nn.init.constant_(self.delta_head.bias, -1.5)

    @staticmethod
    def _stream_to_velocity(psi: torch.Tensor) -> tuple:
        """
        Compute velocity from stream function via discrete curl:
          vx =  ∂ψ/∂y
          vy = -∂ψ/∂x
        """
        # Use constant (zero) padding instead of circular to prevent XLA compile hang
        psi_pad_h = F.pad(psi, [0, 0, 1, 1], mode="constant", value=0.0)
        vx = 0.5 * (psi_pad_h[:, :, 2:, :] - psi_pad_h[:, :, :-2, :])

        psi_pad_w = F.pad(psi, [1, 1, 0, 0], mode="constant", value=0.0)
        vy = -0.5 * (psi_pad_w[:, :, :, 2:] - psi_pad_w[:, :, :, :-2])

        return vx, vy

    @staticmethod
    def _velocity_to_routing(vx: torch.Tensor, vy: torch.Tensor,
                              self_logits: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
        """
        Convert velocity (vx, vy) to self + 8-neighbor routing weights.

        Neighbors: self, up, down, left, right, and four diagonals.
        Neighbor logits come from velocity-direction dot products; the self
        logit is learned directly because velocity alone cannot favor staying.

        Returns: routing weights (B, G, 9, H, W) summing to 1 along dim=2.
        """
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        logits = torch.stack([
            self_logits,
            -vx,
            vx,
            -vy,
            vy,
            (-vx - vy) * inv_sqrt2,
            (-vx + vy) * inv_sqrt2,
            (vx - vy) * inv_sqrt2,
            (vx + vy) * inv_sqrt2,
        ], dim=2) / temperature
        return F.softmax(logits, dim=2)

    def _transport(self, state: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """
        Transport state using self + 8-neighbor weighted gather.

        state:   (B, D, H, W)
        routing: (B, G, 9, H, W) — G groups, each controlling D//G channels

        Returns: transported state (B, D, H, W)
        """
        B, D, H, W = state.shape
        G = self.n_flow_groups
        C = D // G

        # Gather neighbors — reuse padded tensors to reduce F.pad calls
        s_self = state
        s_pad_h = F.pad(state, [0, 0, 1, 1], mode="constant", value=0.0)
        s_up   = s_pad_h[:, :, :-2, :]
        s_down = s_pad_h[:, :, 2:,  :]
        s_pad_w = F.pad(state, [1, 1, 0, 0], mode="constant", value=0.0)
        s_left  = s_pad_w[:, :, :, :-2]
        s_right = s_pad_w[:, :, :, 2:]
        s_pad = F.pad(state, [1, 1, 1, 1], mode="constant", value=0.0)
        s_up_left    = s_pad[:, :, :-2, :-2]
        s_up_right   = s_pad[:, :, :-2, 2:]
        s_down_left  = s_pad[:, :, 2:, :-2]
        s_down_right = s_pad[:, :, 2:, 2:]

        # Efficient single-broadcast weight computation
        scale = self.direction_value_scale.reshape(1, G, C, 9, 1, 1)
        routing_bcast = routing.unsqueeze(2)  # (B, G, 1, 9, H, W)
        combined_weight = (routing_bcast * scale).view(B, D, 9, H, W)

        # [XLA FIX]: Unrolled weighted sum — avoids torch.stack on neighbors
        # which triggers XLA MultiOutputFusion crash. The weight computation
        # above is safe because it uses a single broadcast multiply.
        transported = (
            s_self       * combined_weight[:, :, 0] +
            s_up         * combined_weight[:, :, 1] +
            s_down       * combined_weight[:, :, 2] +
            s_left       * combined_weight[:, :, 3] +
            s_right      * combined_weight[:, :, 4] +
            s_up_left    * combined_weight[:, :, 5] +
            s_up_right   * combined_weight[:, :, 6] +
            s_down_left  * combined_weight[:, :, 7] +
            s_down_right * combined_weight[:, :, 8]
        )
        return transported

    def forward(self, x: torch.Tensor, k_steps: int = 4) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token features
            k_steps: number of transport substeps T

        Returns:
            (B, N, D) updated features
        """
        bsz, num_tokens, d_inner = x.shape
        h = w = int(math.sqrt(num_tokens))

        # --- Initialize paired state ---
        uv = self.state_proj(x)                                     # (B, N, 2D)
        u, v = uv.chunk(2, dim=-1)                                  # each (B, N, D)

        # To spatial: (B, N, D) → (B, D, H, W) using permute + unflatten
        u = u.permute(0, 2, 1).unflatten(2, (h, w))                # (B, D, H, W)
        v = v.permute(0, 2, 1).unflatten(2, (h, w))

        # --- Precompute conditioning once ---

        # Stream function
        psi = self.stream_head(x)                                  # (B, N, G)
        psi = psi.permute(0, 2, 1).unflatten(2, (h, w))            # (B, G, H, W)
        
        self_logits = self.self_route_head(x)
        self_logits = self_logits.permute(0, 2, 1).unflatten(2, (h, w))

        # Velocity from curl of stream function
        vx, vy = self._stream_to_velocity(psi)

        # Routing weights computed ONCE
        routing = self._velocity_to_routing(vx, vy, self_logits)

        # Phase angle computed ONCE
        theta = self.phase_head(x)                                 # (B, N, D)
        theta = theta.permute(0, 2, 1).unflatten(2, (h, w))        # (B, D, H, W)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Gates computed ONCE (Zero-Order Hold discretization)
        delta = F.softplus(self.delta_head(x))                      # (B, N, D)
        retain = torch.exp(-delta).permute(0, 2, 1).unflatten(2, (h, w))
        inject = 1.0 - retain

        # Injection features computed ONCE
        inj = self.inj_proj(x)                                     # (B, N, 2D)
        xu, xv = inj.chunk(2, dim=-1)
        xu = xu.permute(0, 2, 1).unflatten(2, (h, w))
        xv = xv.permute(0, 2, 1).unflatten(2, (h, w))

        # --- Recurrent transport loop ---
        for _ in range(int(k_steps)):
            # Same routing every step
            u_hat = self._transport(u, routing)
            v_hat = self._transport(v, routing)

            # Same rotation every step
            u_rot = cos_t * u_hat - sin_t * v_hat
            v_rot = sin_t * u_hat + cos_t * v_hat

            # Same gates/injection every step
            u = retain * u_rot + inject * xu
            v = retain * v_rot + inject * xv

        # --- Project back to token space ---
        u_out = u.flatten(2).permute(0, 2, 1)                      # (B, N, D)
        v_out = v.flatten(2).permute(0, 2, 1)

        combined = torch.cat([u_out, v_out], dim=-1)                # (B, N, 2D)
        return self.proj_drop(self.out_proj(combined)) + x * self.D


# ---------------------------------------------------------------------------
# Outer Block
# ---------------------------------------------------------------------------

class CharacteristicMambaBlock_V6(nn.Module):
    """Drop-in block wrapping the characteristic transport operator."""

    def __init__(self, d_model: int, expand: int = 2, drop_path: float = 0.0,
                 n_flow_groups: int = 4, proj_drop: float = 0.0):
        super().__init__()
        d_inner = int(expand * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.local_conv2d = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True,
        )
        self.activation = nn.SiLU()
        self.ssm = CharacteristicTransport2D(
            d_model=d_model, expand=expand, n_flow_groups=n_flow_groups,
            proj_drop=proj_drop,
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, k_steps: int = 4) -> torch.Tensor:
        residual = x
        bsz, num_tokens, _ = x.shape
        h = w = int(math.sqrt(num_tokens))

        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)

        # Local preprocessing: (B, N, D) → (B, D, H, W) → conv → back
        u_2d = u.permute(0, 2, 1).unflatten(2, (h, w))
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.flatten(2).permute(0, 2, 1)
        u = self.activation(u)

        y_ssm = self.ssm(u, k_steps=k_steps)
        out = self.proj_drop(self.out_proj(y_ssm * F.silu(z)))
        return residual + self.drop_path(out)


# ---------------------------------------------------------------------------
# Top-level Model
# ---------------------------------------------------------------------------

class CSMamba_V6(nn.Module):
    """
    CS-Mamba V6 — Characteristic Mamba
    ====================================
    Selective 2D characteristic transport with learned divergence-free flow,
    optional phase-rotation coupling, and continuous-time retain/inject gating.
    Linear complexity: O(T · N · D · |neighbors|).
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
        n_flow_groups = getattr(cfg, "n_flow_groups", 4)
        proj_drop_rate = getattr(cfg, "proj_drop", 0.0)
        head_drop_rate = getattr(cfg, "head_drop", 0.0)

        self.k_steps = k_steps
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([
            CharacteristicMambaBlock_V6(
                d_model=d_embed, drop_path=dp_rates[i], n_flow_groups=n_flow_groups,
                proj_drop=proj_drop_rate,
            )
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_embed)
        self.head_drop = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(d_embed, n_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CSMamba_V6 (Characteristic Mamba) %.1fM params, d=%d, layers=%d, K=%d, G=%d",
            n_params / 1e6, d_embed, n_layers, k_steps, n_flow_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            x = layer(x, k_steps=self.k_steps)
        return self.head(self.head_drop(self.final_norm(x).mean(dim=1)))


__all__ = [
    "CharacteristicTransport2D",
    "CharacteristicMambaBlock_V6",
    "CSMamba_V6",
]
