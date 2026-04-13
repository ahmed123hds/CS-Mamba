"""
CS-Mamba PDE Solver — autograd.Function Wrapper (Part 5)
========================================================
Uses the PyTorch reference forward for now. When Triton kernels are validated,
the forward/backward calls get swapped to Triton while the API stays identical.

Usage:
    h_final = cs_scan(h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W)
"""

import torch
from triton_kernels.csma_reference import cs_mamba_forward_reference


class CSScanFunction(torch.autograd.Function):
    """
    Custom autograd function for the CS-Mamba PDE integration.

    Forward:  K-step Euler integration of the thermodynamic PDE
    Backward: Currently uses PyTorch autograd (will be replaced by Triton backward)

    Gradient flow:
        h0, x, delta_s, delta_d, A, B_mat  — all receive gradients
        D_phys                              — receives gradients
        K, H, W                            — integer hyperparams, no gradient
    """

    @staticmethod
    def forward(ctx, h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W):
        """
        Args:
            h0:       (B, N, D, S)   initial state
            x:        (B, N, D)      input (constant across K steps)
            delta_s:  (B, N, D)      self-gating time scale
            delta_d:  (B, N, D)      diffusion-gating time scale
            A:        (D, S)         diagonal decay (negative)
            B_mat:    (B, N, S)      input projection
            D_phys:   (1, D, 1, 1)  diffusivity constant
            K:        int            number of Euler steps
            H, W:     int            spatial grid dimensions

        Returns:
            h_final:  (B, N, D, S)   state after K steps
        """
        # For the autograd-based backward, we need all inputs to require_grad
        # and we use autograd's tape to track gradients. Save for backward.
        ctx.save_for_backward(h0, x, delta_s, delta_d, A, B_mat, D_phys)
        ctx.K = K
        ctx.H = H
        ctx.W = W

        # Run the forward (autograd tracks this for backward)
        h_final, h_saved = cs_mamba_forward_reference(
            h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W
        )

        return h_final

    @staticmethod
    def backward(ctx, grad_output):
        """
        Uses PyTorch autograd recomputation for correctness.
        Will be replaced by Triton backward kernel after validation.
        """
        h0, x, delta_s, delta_d, A, B_mat, D_phys = ctx.saved_tensors
        K, H, W = ctx.K, ctx.H, ctx.W

        # Recompute forward with autograd enabled to get gradients
        h0_r = h0.detach().requires_grad_(True)
        x_r = x.detach().requires_grad_(True)
        ds_r = delta_s.detach().requires_grad_(True)
        dd_r = delta_d.detach().requires_grad_(True)
        A_r = A.detach().requires_grad_(True)
        bm_r = B_mat.detach().requires_grad_(True)
        dp_r = D_phys.detach().requires_grad_(True)

        with torch.enable_grad():
            h_final, _ = cs_mamba_forward_reference(
                h0_r, x_r, ds_r, dd_r, A_r, bm_r, dp_r, K, H, W
            )
            h_final.backward(grad_output)

        return (
            h0_r.grad,     # grad_h0
            x_r.grad,      # grad_x
            ds_r.grad,     # grad_delta_s
            dd_r.grad,     # grad_delta_d
            A_r.grad,      # grad_A
            bm_r.grad,     # grad_B_mat
            dp_r.grad,     # grad_D_phys
            None,          # K (int, no grad)
            None,          # H (int, no grad)
            None,          # W (int, no grad)
        )


def cs_scan(h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W):
    """Convenience wrapper."""
    return CSScanFunction.apply(h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W)


# ── Validation ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CSScanFunction — autograd.Function Validation")
    print("=" * 60)

    torch.manual_seed(42)
    B, D, S, H, W, K = 2, 8, 4, 4, 4, 2
    N = H * W

    h0 = torch.randn(B, N, D, S, dtype=torch.float64, requires_grad=True)
    x  = torch.randn(B, N, D, dtype=torch.float64, requires_grad=True)
    ds = (torch.rand(B, N, D, dtype=torch.float64) * 0.15).requires_grad_(True)
    dd = (torch.rand(B, N, D, dtype=torch.float64) * 0.15).requires_grad_(True)
    A  = (-torch.rand(D, S, dtype=torch.float64).abs()).requires_grad_(True)
    bm = torch.randn(B, N, S, dtype=torch.float64, requires_grad=True)
    dp = (torch.rand(1, D, 1, 1, dtype=torch.float64) * 0.5).requires_grad_(True)

    # Forward
    h_out = cs_scan(h0, x, ds, dd, A, bm, dp, K, H, W)
    print(f"Forward OK: {tuple(h_out.shape)}")

    # Backward
    loss = h_out.sum()
    loss.backward()
    print(f"Backward OK")
    print(f"  grad_h0:      {h0.grad.norm():.6f}")
    print(f"  grad_x:       {x.grad.norm():.6f}")
    print(f"  grad_delta_s: {ds.grad.norm():.6f}")
    print(f"  grad_delta_d: {dd.grad.norm():.6f}")
    print(f"  grad_A:       {A.grad.norm():.6f}")
    print(f"  grad_B_mat:   {bm.grad.norm():.6f}")
    print(f"  grad_D_phys:  {dp.grad.norm():.6f}")

    # Gradcheck
    print("\n--- torch.autograd.gradcheck ---")
    from torch.autograd import gradcheck
    h0_gc = h0.detach().clone().requires_grad_(True)
    x_gc = x.detach().clone().requires_grad_(True)
    ds_gc = ds.detach().clone().requires_grad_(True)
    dd_gc = dd.detach().clone().requires_grad_(True)
    A_gc = A.detach().clone().requires_grad_(True)
    bm_gc = bm.detach().clone().requires_grad_(True)
    dp_gc = dp.detach().clone().requires_grad_(True)

    try:
        ok = gradcheck(
            CSScanFunction.apply,
            (h0_gc, x_gc, ds_gc, dd_gc, A_gc, bm_gc, dp_gc, K, H, W),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        )
        print(f"✓ gradcheck PASSED")
    except Exception as e:
        print(f"✗ gradcheck FAILED: {e}")
