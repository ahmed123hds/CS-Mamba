import torch
import numpy as np
import matplotlib.pyplot as plt

def test():
    from visualize_erf import compute_erf, CSMamba_V1, CSMamba_V3, CSMamba_V4
    for cls in [CSMamba_V1, CSMamba_V3, CSMamba_V4]:
        erf = compute_erf(cls, 'cuda')
        print(f"Model: {cls.__name__}")
        print(f"Max: {erf.max()}, Min: {erf.min()}, Mean: {erf.mean()}")
        print(f"Non-zeros > 0.01: {np.sum(erf > 0.01)}")

if __name__ == "__main__":
    test()
