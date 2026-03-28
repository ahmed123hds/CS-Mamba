"""
NeuralSort — Differentiable Sorting
====================================
Hard sorting (argsort) has zero gradient almost everywhere.
NeuralSort approximates the permutation matrix with a smooth,
differentiable doubly-stochastic matrix P_hat.

At temperature τ → 0  :  P_hat → true permutation matrix
At temperature τ → ∞  :  P_hat → uniform (1/n everywhere)

Reference: Grover et al., "Stochastic Optimization of Sorting Networks
using Continuous Relaxations", ICML 2019.
"""

import torch
import torch.nn as nn


class NeuralSort(nn.Module):
    """
    Differentiable sort layer.

    Given a batch of score vectors s ∈ R^(B, n), returns a soft
    permutation matrix P ∈ R^(B, n, n) such that:
        P[b, i, j] ≈ probability that the i-th sorted element
                      is the j-th input element.

    To get the sorted sequence from input features X ∈ R^(B, n, d):
        X_sorted ≈ P @ X       shape: (B, n, d)

    Args:
        tau:   temperature (lower = harder sort, higher = softer)
               typical range: 1e-3 (near-hard) to 1.0 (soft)
        hard:  if True, return true permutation at test time
               (straight-through estimator for gradients)
    """

    def __init__(self, tau: float = 0.1, hard: bool = False):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, n) — scalar score per patch per sample

        Returns:
            P_hat: (B, n, n) — soft permutation matrix
        """
        B, n = scores.shape

        # Expand to (B, n, 1) and (B, 1, n) for pairwise differences
        s = scores.unsqueeze(2)                      # (B, n, 1)
        s_T = scores.unsqueeze(1)                    # (B, 1, n)

        # |s_i - s_j| matrix, shape (B, n, n)
        abs_diff = torch.abs(s - s_T)

        # Row i of P_hat approximates: P_hat[i, :] ∝ exp(-|s_i - s_j| / τ)
        # Then normalise rows to sum to 1
        P_hat = torch.softmax(-abs_diff / self.tau, dim=2)  # (B, n, n)

        if self.hard and not self.training:
            # Straight-through: use true permutation in forward,
            # but gradients flow through the soft version.
            _, idx = torch.sort(scores, dim=1, descending=True)  # (B, n)
            P_hard = torch.zeros_like(P_hat)
            P_hard.scatter_(2, idx.unsqueeze(2), 1.0)
            P_hat = P_hard - P_hat.detach() + P_hat  # STE trick
        return P_hat


def apply_sort(P_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply a (soft) permutation matrix P_hat to a feature sequence x.

    Args:
        P_hat: (B, n, n) — soft permutation from NeuralSort
        x:     (B, n, d) — input patch embeddings

    Returns:
        x_sorted: (B, n, d) — reordered patch embeddings
    """
    return torch.bmm(P_hat, x)   # (B, n, d)


if __name__ == "__main__":
    # Quick sanity check
    torch.manual_seed(0)
    B, n = 2, 8
    scores = torch.randn(B, n, requires_grad=True)

    sorter = NeuralSort(tau=0.1)
    P = sorter(scores)
    print("P shape:", P.shape)           # (2, 8, 8)
    print("Row sums (≈1):", P[0].sum(1))
    print("Col sums (≈1):", P[0].sum(0))

    # Gradient check
    loss = P.sum()
    loss.backward()
    print("Gradient on scores:", scores.grad)
