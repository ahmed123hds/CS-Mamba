"""Quick numerical checks for the V5 PDE block."""

import torch
from continuous_spatial_mamba_v5 import ExactReactionDiffusion2D


def main():
    torch.manual_seed(0)
    block = ExactReactionDiffusion2D(d_model=8, expand=2)
    block.eval()

    bsz, h, w, d = 2, 8, 8, 16
    x = torch.randn(bsz, h * w, d)

    # Disable reaction to test norm preservation of the diffusion substep.
    with torch.no_grad():
        block.reaction_alpha_raw.zero_()
        block.reaction_beta_raw.fill_(-20.0)  # beta ~ 0

        pooled = x.mean(dim=1)
        gamma = torch.nn.functional.softplus(block.diffusion_gate(pooled) + block.log_diffusion_base)
        gamma = gamma.clamp(max=block.max_diffusion).to(torch.float32)
        lap = block._laplacian_eigs(h, w, x.device)

        psi0 = torch.complex(
            x.to(torch.float32).transpose(1, 2).reshape(bsz, d, h, w),
            torch.zeros(bsz, d, h, w, dtype=torch.float32),
        )
        e0 = (psi0.real.square() + psi0.imag.square()).sum(dim=(-1, -2))
        psi1 = block._exact_diffusion_step(psi0, gamma, lap, dt=0.25)
        e1 = (psi1.real.square() + psi1.imag.square()).sum(dim=(-1, -2))
        rel_err = ((e1 - e0).abs() / e0.clamp_min(1e-8)).max().item()
        print(f"max relative energy error after exact diffusion step: {rel_err:.3e}")


if __name__ == "__main__":
    main()
