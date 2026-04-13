"""
CS-Mamba PDE Solver — Triton Forward Kernel (Part 3)
=====================================================
Fused K-step Euler integration with 5-point Laplacian stencil.

Each Triton program processes one (batch, channel) slice of the
S-collapsed hidden state of shape (H, W).

The Laplacian operates on h_collapsed = h.sum(dim=-1) reshaped to (B, D, H, W).
The kernel computes the Laplacian contribution and returns it.
The S-dimension broadcasting is handled OUTSIDE the kernel in Python.

Architecture decision: Since the full PDE step involves (B, N, D, S) tensors
and the Laplacian only operates on the (B, D, H, W) S-collapsed projection,
the Triton kernel computes ONLY the Laplacian portion. The pointwise Mamba
decay and the Euler accumulation are fused in a second kernel or done in
PyTorch. This keeps the kernel simple and numerically validatable.
"""

import torch
import triton
import triton.language as tl


# ── Triton kernel: fused K-step Laplacian with Neumann BC ─────

@triton.jit
def _laplacian_neumann_kernel(
    # Pointers
    H_IN_ptr,        # input:  (B*D, H, W) contiguous
    LAP_OUT_ptr,     # output: (B*D, H, W) contiguous
    # Dimensions
    H: tl.constexpr,
    W: tl.constexpr,
):
    """
    Compute the 5-point Laplacian with Neumann (zero-flux) boundary conditions.

    Each program handles one (batch, channel) slice = one (H, W) tile.

    Neumann BC: at boundaries, the ghost cell equals the boundary cell.
    This means the stencil contribution from out-of-bounds neighbors is 0
    (since ghost - center = 0), reducing the effective -4 coefficient.

    Implementation: we clamp neighbor indices to [0, H-1] × [0, W-1],
    which replicates the boundary value — equivalent to replicate padding.
    """
    # Which (batch, channel) slice are we?
    bd_idx = tl.program_id(0)

    # Base pointer for this slice
    base = bd_idx * H * W

    # Process each spatial position
    for i in range(H):
        for j in range(W):
            center_off = base + i * W + j
            center_val = tl.load(H_IN_ptr + center_off)

            # Clamped neighbor indices (Neumann BC via index clamping)
            i_top = tl.maximum(i - 1, 0)
            i_bot = tl.minimum(i + 1, H - 1)
            j_left = tl.maximum(j - 1, 0)
            j_right = tl.minimum(j + 1, W - 1)

            top_val   = tl.load(H_IN_ptr + base + i_top * W + j)
            bot_val   = tl.load(H_IN_ptr + base + i_bot * W + j)
            left_val  = tl.load(H_IN_ptr + base + i * W + j_left)
            right_val = tl.load(H_IN_ptr + base + i * W + j_right)

            lap_val = top_val + bot_val + left_val + right_val - 4.0 * center_val
            tl.store(LAP_OUT_ptr + center_off, lap_val)


@triton.jit
def _fused_euler_step_kernel(
    # State pointers (read/write)
    H_STATE_ptr,     # (B*D*S, H*W) — full hidden state, flattened per-slice
    # Laplacian result (read-only, computed on collapsed state)
    LAP_ptr,         # (B*D, H*W)  — Laplacian of h.sum(dim=-1)
    # Input parameters (read-only, constant across K steps)
    DELTA_S_ptr,     # (B*N*D,) — flattened delta_self
    DELTA_D_ptr,     # (B*N*D,) — flattened delta_diff
    A_ptr,           # (D*S,) — flattened A matrix
    BX_ptr,          # (B*N*D*S,) — precomputed B⊗x = einsum(x, B_mat)
    D_PHYS_ptr,      # (D,) — diffusivity per channel
    # Scalars
    dt: tl.constexpr,
    # Dimensions
    B_dim: tl.constexpr,
    N: tl.constexpr,
    D_dim: tl.constexpr,
    S_dim: tl.constexpr,
):
    """
    Fused Euler step: h = h + dt * [delta_s * (A*h + BX) + delta_d * D_phys * lap_h]

    Each program handles one (batch, spatial_pos, channel) = one element
    across all S states. This allows vectorized S-dimension processing.

    Grid: (B * N * D,)
    """
    bnd_idx = tl.program_id(0)

    # Decode batch, spatial, channel indices
    b = bnd_idx // (N * D_dim)
    nd_rem = bnd_idx % (N * D_dim)
    n = nd_rem // D_dim
    d = nd_rem % D_dim

    # Load scalars for this (b, n, d)
    ds_val = tl.load(DELTA_S_ptr + bnd_idx)
    dd_val = tl.load(DELTA_D_ptr + bnd_idx)
    d_phys_val = tl.load(D_PHYS_ptr + d)

    # Laplacian value for this (b, d, spatial_pos)
    # LAP has shape (B*D, H*W), index = (b*D + d)*N + n
    lap_idx = (b * D_dim + d) * N + n
    lap_val = tl.load(LAP_ptr + lap_idx)

    # Iterate over S dimension
    s_offsets = tl.arange(0, S_dim)

    # H_STATE shape: (B, N, D, S), stored as (B*N*D*S,)
    h_base = bnd_idx * S_dim
    h_vals = tl.load(H_STATE_ptr + h_base + s_offsets)

    # A shape: (D, S), index = d * S + s
    a_vals = tl.load(A_ptr + d * S_dim + s_offsets)

    # BX shape: (B, N, D, S) = (B*N*D*S,)
    bx_vals = tl.load(BX_ptr + h_base + s_offsets)

    # Force 1: delta_self * (A * h + BX)
    force_1 = ds_val * (a_vals * h_vals + bx_vals)

    # Force 2: delta_diff * D_phys * laplacian (broadcast across S)
    force_2 = dd_val * d_phys_val * lap_val

    # Euler step
    h_new = h_vals + dt * (force_1 + force_2)

    tl.store(H_STATE_ptr + h_base + s_offsets, h_new)


# ── Python launcher ───────────────────────────────────────────

def triton_laplacian_neumann(h_2d: torch.Tensor) -> torch.Tensor:
    """
    Compute 5-point Laplacian with Neumann BC using Triton.

    Args:
        h_2d: (B, C, H, W) contiguous float32 tensor
    Returns:
        lap:  (B, C, H, W) contiguous float32 tensor
    """
    B, C, H, W = h_2d.shape
    h_flat = h_2d.contiguous().view(B * C, H, W)
    lap_out = torch.empty_like(h_flat)

    grid = (B * C,)
    _laplacian_neumann_kernel[grid](h_flat, lap_out, H=H, W=W)

    return lap_out.view(B, C, H, W)


def triton_euler_step(
    h: torch.Tensor,        # (B, N, D, S)
    lap_h: torch.Tensor,    # (B, D, H, W) — Laplacian of h.sum(dim=-1)
    delta_s: torch.Tensor,  # (B, N, D)
    delta_d: torch.Tensor,  # (B, N, D)
    A: torch.Tensor,        # (D, S)
    bx: torch.Tensor,       # (B, N, D, S) — precomputed einsum(x, B_mat)
    D_phys: torch.Tensor,   # (D,) or (1, D, 1, 1)
    dt: float,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    One fused Euler step using Triton.

    Args/Returns: h is modified in-place and returned.
    """
    B_val, N, D_dim, S_dim = h.shape
    assert N == H * W

    # Flatten D_phys
    d_phys_flat = D_phys.view(-1).contiguous()  # (D,)

    # Reshape lap_h from (B, D, H, W) to (B*D, N)
    lap_flat = lap_h.reshape(B_val * D_dim, N).contiguous()

    grid = (B_val * N * D_dim,)

    _fused_euler_step_kernel[grid](
        h.contiguous(),
        lap_flat,
        delta_s.contiguous().view(-1),
        delta_d.contiguous().view(-1),
        A.contiguous().view(-1),
        bx.contiguous().view(-1),
        d_phys_flat,
        dt=dt,
        B_dim=B_val,
        N=N,
        D_dim=D_dim,
        S_dim=S_dim,
    )
    return h


def triton_cs_mamba_forward(
    h0: torch.Tensor,       # (B, N, D, S)
    x: torch.Tensor,        # (B, N, D)
    delta_s: torch.Tensor,  # (B, N, D)
    delta_d: torch.Tensor,  # (B, N, D)
    A: torch.Tensor,        # (D, S)
    B_mat: torch.Tensor,    # (B, N, S)
    D_phys: torch.Tensor,   # (1, D, 1, 1) or scalar
    K: int,
    H: int,
    W: int,
) -> tuple:
    """
    Full K-step Triton forward for CS-Mamba.

    Returns:
        h_final : (B, N, D, S)
        h_saved : list of K tensors for backward
    """
    B_val, N, D_dim, S_dim = h0.shape
    dt = 1.0 / K

    # Precompute BX = einsum('bnd,bns->bnds', x, B_mat) once
    bx = torch.einsum('bnd,bns->bnds', x, B_mat)

    # D_phys flat
    d_phys_flat = D_phys.view(-1)  # (D,)

    h = h0.clone().contiguous()
    h_saved = []

    for k in range(K):
        h_saved.append(h.clone())

        # 1. Collapse S, reshape to 2D, compute Laplacian
        h_collapsed = h.sum(dim=-1)                              # (B, N, D)
        h_2d = h_collapsed.transpose(1, 2).reshape(B_val, D_dim, H, W).contiguous()
        lap_h = triton_laplacian_neumann(h_2d)                   # (B, D, H, W)

        # 2. Fused Euler step (in-place)
        triton_euler_step(h, lap_h, delta_s, delta_d, A, bx, d_phys_flat, dt, H, W)

    return h, h_saved


# ── Validation against PyTorch reference ──────────────────────

if __name__ == "__main__":
    from triton_kernels.csma_reference import (
        cs_mamba_forward_reference,
        laplacian_2d_neumann,
        test_mass_preservation,
    )

    print("=" * 60)
    print("Triton Forward Kernel — Validation")
    print("=" * 60)

    # ── Part 8: Mass preservation via Triton Laplacian ────────
    print("\n--- Mass Preservation (Triton Laplacian) ---")
    h_test = torch.randn(2, 32, 16, 16, device='cuda', dtype=torch.float64)
    lap_triton = triton_laplacian_neumann(h_test)
    mass_change = lap_triton.sum(dim=[-2, -1])
    max_dev = mass_change.abs().max().item()
    status = "✓" if max_dev < 1e-8 else "✗"
    print(f"{status} Mass preservation: max deviation = {max_dev:.2e}")

    # ── Laplacian comparison ──────────────────────────────────
    print("\n--- Laplacian: Triton vs PyTorch Reference ---")
    lap_ref = laplacian_2d_neumann(h_test)
    max_err = (lap_triton - lap_ref).abs().max().item()
    status = "✓" if max_err < 1e-6 else "✗"
    print(f"{status} Max absolute error: {max_err:.2e}")

    # ── Full forward comparison ───────────────────────────────
    print("\n--- Full Forward: Triton vs PyTorch Reference ---")
    for K in [1, 2, 3]:
        torch.manual_seed(42)
        B, D, S, H, W = 2, 32, 16, 16, 16
        N = H * W

        h0 = torch.randn(B, N, D, S, device='cuda', dtype=torch.float32)
        x  = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
        ds = torch.rand(B, N, D, device='cuda', dtype=torch.float32) * 0.15
        dd = torch.rand(B, N, D, device='cuda', dtype=torch.float32) * 0.15
        A  = -torch.rand(D, S, device='cuda', dtype=torch.float32).abs()
        bm = torch.randn(B, N, S, device='cuda', dtype=torch.float32)
        dp = torch.rand(1, D, 1, 1, device='cuda', dtype=torch.float32) * 0.5

        h_triton, _ = triton_cs_mamba_forward(h0, x, ds, dd, A, bm, dp, K, H, W)
        h_ref, _ = cs_mamba_forward_reference(h0, x, ds, dd, A, bm, dp, K, H, W)

        max_err = (h_triton - h_ref).abs().max().item()
        rel_err = max_err / (h_ref.abs().max().item() + 1e-12)
        status = "✓" if max_err < 1e-4 else "✗"
        print(f"  {status} K={K}: max_abs_err={max_err:.2e}, rel_err={rel_err:.2e}")

    # ── Edge cases ────────────────────────────────────────────
    print("\n--- Edge Cases ---")
    for H, W in [(15, 13), (1, 16), (16, 1), (3, 3)]:
        torch.manual_seed(42)
        N = H * W
        h0 = torch.randn(2, N, 8, 4, device='cuda', dtype=torch.float32)
        x  = torch.randn(2, N, 8, device='cuda', dtype=torch.float32)
        ds = torch.rand(2, N, 8, device='cuda', dtype=torch.float32) * 0.15
        dd = torch.rand(2, N, 8, device='cuda', dtype=torch.float32) * 0.15
        A  = -torch.rand(8, 4, device='cuda', dtype=torch.float32).abs()
        bm = torch.randn(2, N, 4, device='cuda', dtype=torch.float32)
        dp = torch.rand(1, 8, 1, 1, device='cuda', dtype=torch.float32) * 0.5

        h_triton, _ = triton_cs_mamba_forward(h0, x, ds, dd, A, bm, dp, 2, H, W)
        h_ref, _ = cs_mamba_forward_reference(h0, x, ds, dd, A, bm, dp, 2, H, W)
        max_err = (h_triton - h_ref).abs().max().item()
        status = "✓" if max_err < 1e-4 else "✗"
        print(f"  {status} H={H}, W={W}: max_err={max_err:.2e}")
