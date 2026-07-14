import torch
import numpy as np
from .utils import get_mgrid


def compute_boundaries(NN, domain: torch.Tensor, grid_size=200):
    """
    Compute classification map and decision boundary contour.
    Args:
        NN: SigmaCamModel wrapper
        domain: Tensor of shape (M, D) describing polygon of interest
    Returns:
        regions: dict with keys 'X', 'Y', 'Z' arrays for contourf
        decision_boundary: list of contour lines at threshold
    """
    # Prepare grid
    pts_nd, (X, Y, mask) = get_mgrid(domain, grid_size=grid_size)
    # Predict
    outputs = NN.predict(pts_nd).cpu().numpy().reshape(-1)
    # Create full Z with NaNs
    Z = np.full(X.shape, np.nan)
    Z_flat = np.full(X.size, np.nan)
    Z_flat[mask.ravel()] = outputs
    Z = Z_flat.reshape(X.shape)
    # Threshold at 0.5 to find boundary
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=[0.5], colors='red')
    # Extract contour segments. `allsegs` works on every matplotlib version;
    # the old `cs.collections` API was deprecated in 3.8 and removed in 3.10.
    lines = [seg for level_segs in cs.allsegs for seg in level_segs if len(seg)]
    plt.close(fig)
    regions = {'X': X, 'Y': Y, 'Z': Z}
    return regions, lines
