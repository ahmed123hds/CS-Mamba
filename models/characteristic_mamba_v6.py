"""
Characteristic Mamba V6 — Selective 2D Characteristic Transport
================================================================
Key ideas (from char_mamba_methodology.pdf):
  1. Learn a per-location transport direction via a stream function ψ
     whose curl gives a divergence-free velocity field.
  2. Convert velocity to 5-neighbor routing weights via softmax
     (4 neighbors + self/stay path with learned bias).
  3. Transport paired state (U, V) by weighted neighbor gathering.
  4. Optionally rotate the (U, V) pair by a learned phase angle Θ
     to retain V4's oscillatory channel coupling intuition.
  5. Update state via selective Mamba-style retention/injection gates.

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
      1. Predicts a stream function ψ → derives divergence-free velocity (vx, vy)
      2. Converts velocity to 4-neighbor routing weights
      3. Transports paired state (U, V) via weighted neighbor gather
      4. Applies optional phase rotation on (U, V) pairs
      5. Updates state via selective retention (A) and injection (B) gates

    All operations use real-valued tensors with standard ops (F.conv2d,
    F.pad, softmax) — no FFT, no complex dtype, fully XLA/TPU safe.

    Key design choices:
      - 5-neighbor routing: 4 spatial neighbors + self/stay path.
        At zero velocity, the self-logit dominates so patches stay in place
        instead of destructively averaging neighbors.
      - Configurable temperature for sharper routing decisions.
      - Phase rotation is optional and disabled by default.
    """

    def __init__(self, d_model: int, expand: int = 2, n_flow_groups: int = 4,
                 routing_temperature: float = 0.5, use_phase_rotation: bool = False):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.n_flow_groups = n_flow_groups
        self.routing_temperature = routing_temperature
        self.use_phase_rotation = use_phase_rotation
        d_per_group = d_inner // n_flow_groups

        # --- Stream function head (predicts scalar ψ per flow group) ---
        # ψ is predicted per-group; velocity = curl(ψ) via finite differences
        self.stream_head = nn.Sequential(
            nn.Linear(d_inner, d_inner, bias=True),
            nn.SiLU(),
            nn.Linear(d_inner, n_flow_groups, bias=True),
        )

        # --- Phase rotation head (optional, per-channel angle Θ) ---
        if self.use_phase_rotation:
            self.phase_head = nn.Linear(d_inner, d_inner, bias=True)
            nn.init.zeros_(self.phase_head.weight)
            nn.init.zeros_(self.phase_head.bias)

        # --- Self/stay routing bias (learned per flow group) ---
        # Initialized positive so zero velocity → strong self-retention
        self.self_routing_bias = nn.Parameter(torch.full((1, n_flow_groups, 1, 1), 2.0))

        # --- Selective gates ---
        self.gate_a_head = nn.Linear(d_inner, d_inner, bias=True)  # retention
        self.gate_b_head = nn.Linear(d_inner, d_inner, bias=True)  # injection

        # --- Injection features ---
        self.inj_proj = nn.Linear(d_inner, d_inner * 2, bias=False)

        # --- State initialization (project input to paired state) ---
        self.state_proj = nn.Linear(d_inner, d_inner * 2, bias=False)

        # --- Output: project (U, V) back to d_inner + skip ---
        self.out_proj = nn.Linear(d_inner * 2, d_inner, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Init gates to be mildly retentive at start
        nn.init.constant_(self.gate_a_head.bias, 1.5)   # sigmoid(1.5) ≈ 0.82
        nn.init.constant_(self.gate_b_head.bias, -1.5)   # sigmoid(-1.5) ≈ 0.18

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

    def _velocity_to_routing(self, vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
        """
        Convert velocity (vx, vy) to 5-neighbor routing weights via softmax.

        Neighbors: self, up (-y), down (+y), left (-x), right (+x)
        The self/stay path has a learned bias so zero velocity means "stay".

        Returns: routing weights (B, G, 5, H, W) summing to 1 along dim=2.
        """
        tau = self.routing_temperature
        # Neighbor direction vectors: (dy, dx)
        # up=(-1,0), down=(+1,0), left=(0,-1), right=(0,+1)
        logits_up    = -vx / tau   # dot with (-1, 0)
        logits_down  =  vx / tau   # dot with (+1, 0)
        logits_left  = -vy / tau   # dot with (0, -1)
        logits_right =  vy / tau   # dot with (0, +1)

        # Self/stay logit: learned bias, expanded to match spatial dims
        logits_self = self.self_routing_bias.expand_as(vx)  # (B, G, H, W)

        # Stack and softmax: (B, G, 5, H, W) — self is index 0
        logits = torch.stack([logits_self, logits_up, logits_down, logits_left, logits_right], dim=2)
        return F.softmax(logits, dim=2)

    def _transport(self, state: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """
        Transport state using 5-neighbor weighted gather (4 spatial + self).

        state: (B, D, H, W)
        routing: (B, G, 5, H, W) — G groups, each controlling D//G channels
                 index 0 = self, 1 = up, 2 = down, 3 = left, 4 = right

        Returns: transported state (B, D, H, W)
        """
        B, D, H, W = state.shape
        G = self.n_flow_groups
        C = D // G  # channels per group

        # Gather neighbors via constant padding
        s_up    = F.pad(state, [0, 0, 1, 1], mode="constant", value=0.0)[:, :, :-2, :]
        s_down  = F.pad(state, [0, 0, 1, 1], mode="constant", value=0.0)[:, :, 2:,  :]
        s_left  = F.pad(state, [1, 1, 0, 0], mode="constant", value=0.0)[:, :, :, :-2]
        s_right = F.pad(state, [1, 1, 0, 0], mode="constant", value=0.0)[:, :, :, 2:]

        # Stack: self + 4 neighbors → (B, D, 5, H, W)
        neighbors = torch.stack([state, s_up, s_down, s_left, s_right], dim=2)

        # Reshape for broadcasting
        # neighbors: (B, G, C, 5, H, W)
        neighbors = neighbors.view(B, G, C, 5, H, W)
        # routing: (B, G, 1, 5, H, W)
        routing_bcast = routing.unsqueeze(2)

        # Weighted sum: (B, G, C, H, W) → (B, D, H, W)
        transported = (neighbors * routing_bcast).sum(dim=3).view(B, D, H, W)
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

        # --- Precompute conditioning (stays fixed across substeps) ---
        # Stream function
        psi = self.stream_head(x)                                   # (B, N, G)
        psi = psi.permute(0, 2, 1).unflatten(2, (h, w))            # (B, G, H, W)

        # Velocity from curl of stream function
        vx, vy = self._stream_to_velocity(psi)

        # Routing weights (5-neighbor softmax: self + 4 spatial)
        routing = self._velocity_to_routing(vx, vy)                 # (B, G, 5, H, W)

        # Phase angle (only if enabled)
        if self.use_phase_rotation:
            theta = self.phase_head(x)                              # (B, N, D)
            theta = theta.permute(0, 2, 1).unflatten(2, (h, w))    # (B, D, H, W)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
        else:
            cos_t = None
            sin_t = None

        # Selective gates
        gate_a = torch.sigmoid(self.gate_a_head(x))                 # (B, N, D)
        gate_a = gate_a.permute(0, 2, 1).unflatten(2, (h, w))      # (B, D, H, W)

        gate_b = torch.sigmoid(self.gate_b_head(x))
        gate_b = gate_b.permute(0, 2, 1).unflatten(2, (h, w))

        # Injection features
        inj = self.inj_proj(x)                                      # (B, N, 2D)
        xu, xv = inj.chunk(2, dim=-1)
        xu = xu.permute(0, 2, 1).unflatten(2, (h, w))
        xv = xv.permute(0, 2, 1).unflatten(2, (h, w))

        # --- Recurrent transport loop ---
        for _ in range(int(k_steps)):
            # 1. Transport state along learned flow
            u_hat = self._transport(u, routing)
            v_hat = self._transport(v, routing)

            # 2. Phase rotation (optional oscillatory channel coupling)
            if self.use_phase_rotation:
                u_rot = cos_t * u_hat - sin_t * v_hat
                v_rot = sin_t * u_hat + cos_t * v_hat
            else:
                u_rot = u_hat
                v_rot = v_hat

            # 3. Selective recurrence: retain + inject
            u = gate_a * u_rot + gate_b * xu
            v = gate_a * v_rot + gate_b * xv

        # --- Project back to token space ---
        u_out = u.flatten(2).permute(0, 2, 1)                      # (B, N, D)
        v_out = v.flatten(2).permute(0, 2, 1)

        combined = torch.cat([u_out, v_out], dim=-1)                # (B, N, 2D)
        return self.out_proj(combined) + x * self.D


# ---------------------------------------------------------------------------
# Outer Block
# ---------------------------------------------------------------------------

class CharacteristicMambaBlock_V6(nn.Module):
    """Drop-in block wrapping the characteristic transport operator."""

    def __init__(self, d_model: int, expand: int = 2, drop_path: float = 0.0,
                 n_flow_groups: int = 4, routing_temperature: float = 0.5,
                 use_phase_rotation: bool = False):
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
            routing_temperature=routing_temperature,
            use_phase_rotation=use_phase_rotation,
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
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
        out = self.out_proj(y_ssm * F.silu(z))
        return residual + self.drop_path(out)


# ---------------------------------------------------------------------------
# Top-level Model
# ---------------------------------------------------------------------------

class CSMamba_V6(nn.Module):
    """
    CS-Mamba V6 — Characteristic Mamba
    ====================================
    Selective 2D characteristic transport with learned divergence-free flow,
    optional phase-rotation coupling, and Mamba-style retention/injection gates.
    Linear complexity: O(T · N · D · |neighbors|).
    """

    def __init__(self, cfg):
        super().__init__()
        img_size = getattr(cfg, "img_size", getattr(cfg, "canvas_size", 224))
        patch_size = getattr(cfg, "patch_size", 16)
        d_embed = getattr(cfg, "d_embed", 384)
        n_layers = getattr(cfg, "n_mamba_layers", 12)
        n_classes = getattr(cfg, "n_classes", 1000)
        k_steps = getattr(cfg, "K_steps", 2)
        drop_path_rate = getattr(cfg, "drop_path", 0.1)
        n_flow_groups = getattr(cfg, "n_flow_groups", 4)
        routing_temp = getattr(cfg, "routing_temperature", 0.5)
        use_phase = getattr(cfg, "use_phase_rotation", False)

        self.k_steps = k_steps
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([
            CharacteristicMambaBlock_V6(
                d_model=d_embed, drop_path=dp_rates[i], n_flow_groups=n_flow_groups,
                routing_temperature=routing_temp, use_phase_rotation=use_phase,
            )
            for i in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_embed)
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
        return self.head(self.final_norm(x).mean(dim=1))


__all__ = [
    "CharacteristicTransport2D",
    "CharacteristicMambaBlock_V6",
    "CSMamba_V6",
]
