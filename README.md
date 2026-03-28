# CS-Mamba: Continuous Spatial PDE Mamba (ICLR 2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference: ICLR 2026](https://img.shields.io/badge/Conference-ICLR_2026-blue.svg)](https://iclr.cc)

**CS-Mamba** (Continuous Spatial PDE Mamba) is a novel Physics-Informed Neural Network (PINN) architecture that reformulates the Vision Mamba block as a 2-D Thermodynamic Heat Equation integrator. By replacing 1-D causal sequence canning with a continuous-time PDE diffusion engine, CS-Mamba natively solves the "Topological Amnesia" problem inherent in standard Vision SSMs.

---

## 🚀 Key Results
| Model | Params | Tiny-ImageNet (Acc@1) | Gain |
| :--- | :--- | :--- | :--- |
| Hilbert Mamba (1D Scan) | 4.4M | 56.53% | Baseline |
| **CS-Mamba (2D PDE Engine)** | **4.4M** | **63.47%** | **+6.94%** |

---

## 🧪 Core Methodology
CS-Mamba treats the hidden state as a physical field diffusing across the spatial manifold $\Omega$, governed by:

$$\frac{\partial h}{\partial t}(\mathbf{r},t) = \Delta_{\mathrm{self}} \bigl(A\,h + B\,x\bigr) + \Delta_{\mathrm{diff}} \cdot \mathcal{D}\,\nabla^2 h$$

- **Physics-Informed Laplacian:** Hard-coded $\nabla^2$ operator ensures **Feature Mass Preservation** (Sum=0).
- **Dual-Gating Mechanism:** Decoupled control of internal memory ($\Delta_{\mathrm{self}}$) and spatial broadcast ($\Delta_{\mathrm{diff}}$).
- **CFL Stability engine:** Diffusivity $\mathcal{D}$ is strictly bounded to $\le 0.5$ via a sigmoid constraint to guarantee numerical stability.

---

## 🛠️ Project Structure
- `models/continuous_spatial_mamba.py`: Core PDE integration engine and dual-gate logic.
- `train_tiny_imagenet.py`: SOTA training script with Mixup, CutMix, and Gradient Checkpointing.
- `train_pcam.py`: Medical imaging (WSI Proxy) validation script.
- `cs_mamba_paper_sections.tex`: LaTeX source for the Methodology and Theorems sections.

---

## 🏃 Usage
Run the full comparative benchmark on Tiny-ImageNet:
```bash
python train_tiny_imagenet.py --mode all --epochs 100 --batch_size 64
```

---

## 📜 Citation
If you find this work useful, please cite our ICLR 2026 submission:
```bibtex
@article{ahmed2025csmamba,
  title={Continuous Spatial PDE Mamba: Thermodynamic State Space Models for Vision},
  author={Ahmed et al.},
  journal={arXiv preprint (ICLR 2026 Submission)},
  year={2025}
}
```
