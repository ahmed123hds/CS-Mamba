"""
Neural ODE Router
==================
Takes a bag of patch embeddings and outputs a scalar score per patch.
These scores define the scan order: sort patches by score → feed into Mamba.

Architecture:
    eᵢ ∈ R^d  (patch embeddings, one per patch)
        ↓
    Neural ODE:  dh/dt = f_θ(h, t),  h(0) = eᵢ
        ↓
    h_i(T=1) ∈ R^d   (evolved embeddings)
        ↓
    Linear projection → sᵢ ∈ R   (scalar score per patch)
        ↓
    NeuralSort → soft permutation matrix P ∈ R^(n×n)
        ↓
    P @ embeddings  → reordered sequence for Mamba

The ODE function f_θ is a small MLP shared across all patches.
Because it's shared, patches *implicitly* interact through
common dynamics — patches with similar content evolve similarly
and thus get similar scores (i.e., cluster together in the sequence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint_adjoint as odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False

from utils.neural_sort import NeuralSort, apply_sort


class ODEFunc(nn.Module):
    """
    The dynamics function f_θ(h, t) for the Neural ODE.

    We concatenate the time scalar t to the hidden state h so the
    dynamics can be time-aware (non-autonomous system).

    Args:
        d_hidden: dimension of the hidden state (= patch embedding dim)
        d_ff:     feedforward dimension inside the MLP
    """

    def __init__(self, d_hidden: int, d_ff: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden + 1, d_ff),   # +1 for time
            nn.Tanh(),
            nn.Linear(d_ff, d_ff),
            nn.Tanh(),
            nn.Linear(d_ff, d_hidden),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar tensor (current integration time)
            h: (B*n, d_hidden)  — all patches from all batch items flattened

        Returns:
            dh/dt: same shape as h
        """
        # Append time to every hidden state
        t_expand = t.expand(h.shape[0], 1)
        h_t = torch.cat([h, t_expand], dim=-1)
        return self.net(h_t)


class NeuralODERouter(nn.Module):
    """
    Neural ODE Router module.

    Given patch embeddings, evolves them through a Neural ODE and
    outputs a differentiable scan order via NeuralSort.

    Args:
        d_embed:    patch embedding dimension
        d_ff:       feedforward dim inside ODE function
        tau:        NeuralSort temperature (lower → harder sort)
        t_span:     integration interval [t0, t1]
        solver:     ODE solver ('euler', 'rk4', 'dopri5')
        n_steps:    number of fixed steps (only for euler/rk4)
    """

    def __init__(
        self,
        d_embed:  int,
        d_ff:     int   = 64,
        tau:      float = 0.1,
        t_span:   tuple = (0.0, 1.0),
        solver:   str   = 'rk4',
        n_steps:  int   = 10,
    ):
        super().__init__()
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq is required. Install with:\n"
                "  pip install torchdiffeq"
            )

        self.d_embed = d_embed
        self.solver  = solver
        self.n_steps = n_steps

        t0, t1 = t_span
        self.register_buffer('t_span', torch.tensor([t0, t1]))

        self.ode_func   = ODEFunc(d_embed, d_ff)
        self.score_head = nn.Linear(d_embed, 1)   # embed → scalar score
        self.sorter     = NeuralSort(tau=tau)

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, n, d_embed)  — n patches per image

        Returns:
            x_sorted:  (B, n, d_embed)   — patches in learned order
            scores:    (B, n)            — raw scores (for visualisation)
        """
        B, n, d = embeddings.shape

        # Flatten batch and patches for ODE integration: (B*n, d)
        h0 = embeddings.reshape(B * n, d)

        # Integrate ODE from t=0 to t=1
        # odeint returns shape (T, B*n, d); we take the final state [-1]
        solver_options = {}
        if self.solver in ('euler', 'rk4'):
            solver_options['step_size'] = (
                (self.t_span[1] - self.t_span[0]).item() / self.n_steps
            )

        h_final = odeint(
            self.ode_func,
            h0,
            self.t_span,
            method=self.solver,
            options=solver_options if solver_options else None,
        )[-1]                                  # (B*n, d)

        h_final = h_final.reshape(B, n, d)    # (B, n, d)

        # Project to scalar scores: (B, n)
        scores = self.score_head(h_final).squeeze(-1)

        # Differentiable sort → soft permutation
        P_hat    = self.sorter(scores)         # (B, n, n)
        x_sorted = apply_sort(P_hat, embeddings)  # (B, n, d)

        return x_sorted, scores


class FixedRouterHilbert(nn.Module):
    """
    Baseline router using a fixed Hilbert curve scan order.
    No parameters — just reorders patches deterministically.

    Args:
        grid_h, grid_w: patch grid dimensions
    """

    def __init__(self, grid_h: int, grid_w: int):
        super().__init__()
        import numpy as np
        from utils.hilbert import get_hilbert_order
        order = get_hilbert_order(grid_h, grid_w)
        self.register_buffer('order', torch.from_numpy(order))

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            embeddings: (B, n, d)

        Returns:
            x_sorted: (B, n, d) — patches in Hilbert order
            None:     no scores
        """
        return embeddings[:, self.order, :], None
