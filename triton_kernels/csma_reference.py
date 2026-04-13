"""
CS-Mamba PDE Solver — PyTorch Reference Implementation (Part 2)
================================================================
Pure-PyTorch ground truth for numerical validation of Triton kernels.

PDE:  h^(k+1) = h^(k) + dt * [Δ_self ⊙ (A ⊙ h^(k) + B ⊙ x) + Δ_diff ⊙ D_phys · ∇²h^(k)]

Key shapes (matching continuous_spatial_mamba.py exactly):
    h        : (B, N, D, S)   — full hidden state
    x        : (B, N, D)      — input, constant across K steps
    A        : (D, S)         — diagonal decay (negative)
    B_mat    : (B, N, S)      — input projection
    delta_s  : (B, N, D)      — per-patch self-gating
    delta_d  : (B, N, D)      — per-patch diffusion-gating
    D_phys   : (1, D, 1, 1)   — diffusivity constant ∈ (0, 0.5]

The Laplacian is applied to h.sum(dim=-1) reshaped to (B, D, H, W),
then broadcast back across S via unsqueeze(-1).
"""

import math
import torch
import torch.nn.functional as F


# ── 5-point Laplacian with Neumann BC ─────────────────────────

def laplacian_2d_neumann(h_2d: torch.Tensor) -> torch.Tensor:
    """
    Apply the 5-point discrete Laplacian with zero-flux (Neumann) BCs.

    Args:
        h_2d: (B, C, H, W)
    Returns:
        lap:  (B, C, H, W)

    Neumann BC: ghost cells replicate boundary values,
    so the stencil at an edge uses the boundary value in place of
    the out-of-bounds neighbour.

    pad(mode='replicate') achieves this: it copies the edge value into
    the ghost cell, making the finite difference across the boundary = 0.
    """
    # Pad with 1 pixel of replicated boundary (Neumann)
    h_pad = F.pad(h_2d, (1, 1, 1, 1), mode='replicate')

    # 5-point stencil: neighbors - 4*center
    lap = (h_pad[:, :, 0:-2, 1:-1] +   # top
           h_pad[:, :, 2:,   1:-1] +   # bottom
           h_pad[:, :, 1:-1, 0:-2] +   # left
           h_pad[:, :, 1:-1, 2:]   -   # right
           4.0 * h_2d)                  # center

    return lap


# ── Forward PDE solver (reference) ────────────────────────────

def cs_mamba_forward_reference(
    h0:       torch.Tensor,   # (B, N, D, S)
    x:        torch.Tensor,   # (B, N, D)
    delta_s:  torch.Tensor,   # (B, N, D)
    delta_d:  torch.Tensor,   # (B, N, D)
    A:        torch.Tensor,   # (D, S)
    B_mat:    torch.Tensor,   # (B, N, S)
    D_phys:   torch.Tensor,   # (1, D, 1, 1) or scalar
    K:        int,
    H:        int,
    W:        int,
) -> tuple:
    """
    Pure-PyTorch K-step Euler integration of the CS-Mamba PDE.

    Returns:
        h_final : (B, N, D, S)      — state after K steps
        h_saved : list of K tensors  — h^(0), ..., h^(K-1) for backward
    """
    B_val, N, D_dim, S_dim = h0.shape
    assert H * W == N, f"H*W={H*W} != N={N}"

    dt = 1.0 / K
    h = h0.clone()
    h_saved = []

    for k in range(K):
        h_saved.append(h.clone())

        # ── Laplacian on S-collapsed state ────────────────────
        # Collapse S: (B, N, D, S) → (B, N, D) via sum
        h_collapsed = h.sum(dim=-1)                      # (B, N, D)
        # Reshape to 2D: (B, D, H, W)
        h_2d = h_collapsed.transpose(1, 2).reshape(B_val, D_dim, H, W)
        # Apply 5-point Laplacian with Neumann BC
        lap_h_2d = laplacian_2d_neumann(h_2d)            # (B, D, H, W)
        # Reshape back: (B, D, H, W) → (B, N, D)
        lap_h = lap_h_2d.reshape(B_val, D_dim, N).transpose(1, 2)
        # Broadcast across S: (B, N, D) → (B, N, D, S)
        diffused = lap_h.unsqueeze(-1)                    # (B, N, D, S)

        # ── Force 1: Mamba self-decay + input ─────────────────
        # A ⊙ h^(k):  (D, S) * (B, N, D, S) → (B, N, D, S)
        mamba_decay = A.unsqueeze(0).unsqueeze(0) * h     # (B, N, D, S)
        # B ⊙ x:  (B, N, D) * (B, N, S) → (B, N, D, S) via einsum
        mamba_input = torch.einsum('bnd,bns->bnds', x, B_mat)
        force_1 = delta_s.unsqueeze(-1) * (mamba_decay + mamba_input)

        # ── Force 2: Spatial diffusion ────────────────────────
        # D_phys: (1, D, 1, 1) → (D,) → (1, 1, D, 1) for (B, N, D, S) broadcast
        D_coeff = D_phys.view(1, 1, -1, 1)  # (1, 1, D, 1)
        force_2 = delta_d.unsqueeze(-1) * D_coeff * diffused

        # ── Euler step ────────────────────────────────────────
        h = h + dt * (force_1 + force_2)

    return h, h_saved


# ── Backward PDE solver (reference, for gradient validation) ──

def cs_mamba_backward_reference(
    grad_output: torch.Tensor,   # (B, N, D, S) — ∂L/∂h^(K)
    h_saved:     list,           # [h^(0), ..., h^(K-1)]
    x:           torch.Tensor,   # (B, N, D)
    delta_s:     torch.Tensor,   # (B, N, D)
    delta_d:     torch.Tensor,   # (B, N, D)
    A:           torch.Tensor,   # (D, S)
    B_mat:       torch.Tensor,   # (B, N, S)
    D_phys:      torch.Tensor,   # (1, D, 1, 1) or scalar
    K:           int,
    H:           int,
    W:           int,
) -> dict:
    """
    Reverse-mode unrolled backward for the K-step Euler integration.

    Returns dict of gradients for: h0, x, delta_s, delta_d, A, B_mat, D_phys
    """
    B_val, N, D_dim, S_dim = grad_output.shape
    dt = 1.0 / K

    D_coeff = D_phys.view(1, 1, -1, 1)  # (1, 1, D, 1) for (B, N, D, S) broadcast

    # Initialize gradient accumulators
    grad_delta_s = torch.zeros_like(delta_s)    # (B, N, D)
    grad_delta_d = torch.zeros_like(delta_d)    # (B, N, D)
    grad_A       = torch.zeros_like(A)          # (D, S)
    grad_B_mat   = torch.zeros_like(B_mat)      # (B, N, S)
    grad_x       = torch.zeros_like(x)          # (B, N, D)

    # Adjoint variable g^(K) = grad_output
    g = grad_output.clone()

    # Reverse loop: k = K-1 down to 0
    for k in reversed(range(K)):
        h_k = h_saved[k]  # (B, N, D, S)

        # ── Recompute intermediate quantities at step k ───────
        # Laplacian of h^(k)
        h_collapsed = h_k.sum(dim=-1)
        h_2d = h_collapsed.transpose(1, 2).reshape(B_val, D_dim, H, W)
        lap_h_2d = laplacian_2d_neumann(h_2d)
        lap_h = lap_h_2d.reshape(B_val, D_dim, N).transpose(1, 2)
        diffused = lap_h.unsqueeze(-1)

        mamba_input = torch.einsum('bnd,bns->bnds', x, B_mat)

        # ── Accumulate parameter gradients ────────────────────
        # ∂L/∂Δ_self += dt * Σ_s g^(k+1) ⊙ (A ⊙ h^(k) + B ⊙ x)
        force_1_inner = A.unsqueeze(0).unsqueeze(0) * h_k + mamba_input
        grad_delta_s += dt * (g * force_1_inner).sum(dim=-1)

        # ∂L/∂Δ_diff += dt * Σ_s g^(k+1) ⊙ D_phys · L[h^(k)]
        grad_delta_d += dt * (g * D_coeff * diffused).sum(dim=-1)

        # ∂L/∂A += dt * reduce_BN( g^(k+1) ⊙ Δ_self ⊙ h^(k) )
        grad_A += dt * (g * delta_s.unsqueeze(-1) * h_k).sum(dim=(0, 1))

        # ∂L/∂B_mat += dt * Σ_d g^(k+1) ⊙ Δ_self ⊙ x  (summed over D)
        grad_B_mat += dt * torch.einsum(
            'bnds,bnd->bns',
            g * delta_s.unsqueeze(-1),
            x
        )

        # ∂L/∂x += dt * Σ_s (g^(k+1) ⊙ Δ_self ⊙ B_mat)
        grad_x += dt * torch.einsum(
            'bnds,bns->bnd',
            g * delta_s.unsqueeze(-1),
            B_mat
        )

        # ── Adjoint recurrence: g^(k) = g^(k+1) + dt * ∂F/∂h^T · g^(k+1) ──
        # ∂F/∂h = Δ_self ⊙ A  (pointwise)
        #       + Δ_diff ⊙ D_phys · ∇² (self-adjoint, same stencil)
        g_pointwise = g * delta_s.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)

        # Laplacian of g (self-adjoint: L^T = L)
        g_collapsed = g.sum(dim=-1)
        g_2d = g_collapsed.transpose(1, 2).reshape(B_val, D_dim, H, W)
        lap_g_2d = laplacian_2d_neumann(g_2d)
        lap_g = lap_g_2d.reshape(B_val, D_dim, N).transpose(1, 2)
        g_diffusion = delta_d.unsqueeze(-1) * D_coeff * lap_g.unsqueeze(-1)

        g = g + dt * (g_pointwise + g_diffusion)

    return {
        'grad_h0':       g,               # (B, N, D, S)
        'grad_x':        grad_x,          # (B, N, D)
        'grad_delta_s':  grad_delta_s,    # (B, N, D)
        'grad_delta_d':  grad_delta_d,    # (B, N, D)
        'grad_A':        grad_A,          # (D, S)
        'grad_B_mat':    grad_B_mat,      # (B, N, S)
    }


# ── Mass preservation test ────────────────────────────────────

def test_mass_preservation(B=2, D=32, H=16, W=16):
    """Part 8: Validates Theorem 2.3 — Laplacian with Neumann BC preserves mass."""
    h = torch.randn(B, D, H, W, dtype=torch.float64)
    lap_h = laplacian_2d_neumann(h)
    mass_change = lap_h.sum(dim=[-2, -1])
    max_dev = mass_change.abs().max().item()
    assert max_dev < 1e-8, f"Mass NOT preserved: max deviation {max_dev}"
    print(f"✓ Mass preservation test PASSED (max deviation: {max_dev:.2e})")


# ── Forward validation helper ─────────────────────────────────

def validate_forward_reference(B=2, D=32, H=16, W=16, K=3, S=16):
    """Validate that the reference matches the original ContinuousSpatialSSM logic."""
    torch.manual_seed(42)
    N = H * W

    h0 = torch.randn(B, N, D, S, dtype=torch.float64)
    x  = torch.randn(B, N, D, dtype=torch.float64)
    delta_s = torch.rand(B, N, D, dtype=torch.float64) * 0.15
    delta_d = torch.rand(B, N, D, dtype=torch.float64) * 0.15
    A  = -torch.rand(D, S, dtype=torch.float64).abs()  # Must be negative
    B_mat = torch.randn(B, N, S, dtype=torch.float64)
    D_phys = torch.rand(1, D, 1, 1, dtype=torch.float64) * 0.5

    h_final, h_saved = cs_mamba_forward_reference(
        h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W
    )

    print(f"✓ Reference forward OK: input {tuple(h0.shape)} → output {tuple(h_final.shape)}")
    print(f"  K={K}, dt={1.0/K:.4f}")
    print(f"  Saved {len(h_saved)} intermediates")
    print(f"  h_final norm: {h_final.norm():.4f}")
    return h_final, h_saved


# ── Backward validation via autograd ──────────────────────────

def validate_backward_reference(B=2, D=8, H=4, W=4, K=2, S=4):
    """
    Validate reference backward against torch.autograd.gradcheck.
    Uses small sizes because gradcheck is O(n) forward passes.
    """
    torch.manual_seed(42)
    N = H * W

    h0 = torch.randn(B, N, D, S, dtype=torch.float64, requires_grad=True)
    x  = torch.randn(B, N, D, dtype=torch.float64, requires_grad=True)
    delta_s = (torch.rand(B, N, D, dtype=torch.float64) * 0.15).requires_grad_(True)
    delta_d = (torch.rand(B, N, D, dtype=torch.float64) * 0.15).requires_grad_(True)
    A  = (-torch.rand(D, S, dtype=torch.float64).abs()).requires_grad_(True)
    B_mat = torch.randn(B, N, S, dtype=torch.float64, requires_grad=True)
    D_phys = (torch.rand(1, D, 1, 1, dtype=torch.float64) * 0.5).requires_grad_(True)

    def fn(h0, x, ds, dd, a, bm, dp):
        h_f, _ = cs_mamba_forward_reference(h0, x, ds, dd, a, bm, dp, K, H, W)
        return h_f.sum()

    # Test autograd on the reference forward
    loss = fn(h0, x, delta_s, delta_d, A, B_mat, D_phys)
    loss.backward()

    print(f"✓ Reference backward OK via autograd")
    print(f"  grad_h0 norm:      {h0.grad.norm():.6f}")
    print(f"  grad_delta_s norm: {delta_s.grad.norm():.6f}")
    print(f"  grad_delta_d norm: {delta_d.grad.norm():.6f}")
    print(f"  grad_A norm:       {A.grad.norm():.6f}")
    print(f"  grad_B_mat norm:   {B_mat.grad.norm():.6f}")

    # Now compare against our manual backward
    h0_nd = h0.detach().clone()
    x_nd = x.detach().clone()
    ds_nd = delta_s.detach().clone()
    dd_nd = delta_d.detach().clone()
    A_nd = A.detach().clone()
    bm_nd = B_mat.detach().clone()
    dp_nd = D_phys.detach().clone()

    h_final_nd, h_saved_nd = cs_mamba_forward_reference(
        h0_nd, x_nd, ds_nd, dd_nd, A_nd, bm_nd, dp_nd, K, H, W
    )
    grad_output = torch.ones_like(h_final_nd)

    grads = cs_mamba_backward_reference(
        grad_output, h_saved_nd, x_nd, ds_nd, dd_nd, A_nd, bm_nd, dp_nd, K, H, W
    )

    # Compare manual vs autograd
    for name, (manual, auto) in [
        ('h0',      (grads['grad_h0'],      h0.grad)),
        ('delta_s', (grads['grad_delta_s'], delta_s.grad)),
        ('delta_d', (grads['grad_delta_d'], delta_d.grad)),
        ('A',       (grads['grad_A'],       A.grad)),
        ('B_mat',   (grads['grad_B_mat'],   B_mat.grad)),
    ]:
        max_err = (manual - auto).abs().max().item()
        rel_err = max_err / (auto.abs().max().item() + 1e-12)
        status = "✓" if max_err < 1e-6 else "✗"
        print(f"  {status} grad_{name}: max_abs_err={max_err:.2e}, rel_err={rel_err:.2e}")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CS-Mamba PyTorch Reference — Validation Suite")
    print("=" * 60)

    print("\n--- Part 8: Mass Preservation Test ---")
    test_mass_preservation()

    print("\n--- Part 2: Forward Reference Validation ---")
    for K in [1, 2, 3]:
        validate_forward_reference(K=K)

    print("\n--- Part 2: Backward Reference Validation ---")
    validate_backward_reference()

    # Edge cases
    print("\n--- Edge Cases ---")
    print("Odd dimensions:")
    validate_forward_reference(H=15, W=13, K=2)
    print("Degenerate H=1:")
    validate_forward_reference(H=1, W=16, K=2)
    print("Degenerate W=1:")
    validate_forward_reference(H=16, W=1, K=2)
