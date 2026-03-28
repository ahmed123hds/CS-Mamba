"""
Hilbert curve scan order for 2D patch grids.
Converts a (rows, cols) grid into a 1D traversal order
that preserves 2D locality — nearby patches in 2D stay
nearby in the 1D sequence.
"""

import numpy as np


def _xy2d(n: int, x: int, y: int) -> int:
    """Convert (x, y) to Hilbert curve distance d in an n×n grid."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def get_hilbert_order(grid_h: int, grid_w: int) -> np.ndarray:
    """
    Returns a 1D array of patch indices sorted by Hilbert curve order.

    For a grid_h × grid_w patch grid, this returns the traversal
    order that maximises 2D locality in the 1D sequence.

    If grid dimensions are not a power of 2, we embed into the next
    power-of-2 grid, compute the curve, then filter back.

    Args:
        grid_h: number of patch rows
        grid_w: number of patch columns

    Returns:
        np.ndarray of shape (grid_h * grid_w,) — indices into the
        flattened patch list, ordered by Hilbert distance.
    """
    # Use largest dimension, round up to power of 2
    n = max(grid_h, grid_w)
    n = int(2 ** np.ceil(np.log2(max(n, 1))))

    # Compute (distance, flat_index) for every valid grid cell
    pairs = []
    for row in range(grid_h):
        for col in range(grid_w):
            d = _xy2d(n, col, row)
            flat_idx = row * grid_w + col
            pairs.append((d, flat_idx))

    pairs.sort(key=lambda p: p[0])
    return np.array([p[1] for p in pairs], dtype=np.int64)


if __name__ == "__main__":
    order = get_hilbert_order(4, 4)
    print("Hilbert order for 4×4 grid:")
    grid = np.zeros((4, 4), dtype=int)
    for step, idx in enumerate(order):
        r, c = divmod(idx, 4)
        grid[r, c] = step
    print(grid)
