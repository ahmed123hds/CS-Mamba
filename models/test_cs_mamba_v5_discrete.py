"""Quick numerical checks for the real-valued V5 PDE block."""

from pathlib import Path

import torch
from continuous_spatial_mamba_v5 import RealSymplecticReactionDiffusion2D


def main():
    torch.manual_seed(0)
    block = RealSymplecticReactionDiffusion2D(d_model=8, expand=2)
    block.eval()

    bsz, h, w, d = 2, 8, 8, 16
    x = torch.randn(bsz, h * w, d)

    with torch.no_grad():
        # Turn reaction off to isolate the symplectic linear flow.
        block.reaction_alpha_raw.zero_()
        block.reaction_beta_raw.fill_(-20.0)  # beta ~ 0 after softplus

        gamma, alpha, beta = block._reaction_params(x)
        u = x.transpose(1, 2).reshape(bsz, d, h, w)
        v = torch.zeros_like(u)
        e0 = (u.square() + v.square()).sum(dim=(-1, -2, -3))

        for _ in range(50):
            u, v = block._leapfrog_hamiltonian_step(u, v, gamma, dt=0.25)

        e1 = (u.square() + v.square()).sum(dim=(-1, -2, -3))
        rel_drift = ((e1 - e0).abs() / e0.clamp_min(1e-8)).max().item()
        print(f"max relative norm drift after 50 linear steps: {rel_drift:.3e}")

    source = Path(__file__).with_name("continuous_spatial_mamba_v5.py").read_text()
    banned = ["torch.fft", "torch.complex"]
    for token in banned:
        assert token not in source, f"Found banned token in source: {token}"
    print("source check passed: no torch.fft / torch.complex")


if __name__ == "__main__":
    main()
