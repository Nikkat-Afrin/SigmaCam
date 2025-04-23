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
    import matplotlib.pyplot as plt
    cs = plt.contour(X, Y, Z, levels=[0.5], colors='red')
    # Extract lines
    lines = []
    for collection in cs.collections:
        for path in collection.get_paths():
            lines.append(path.vertices)
    plt.close()
    regions = {'X': X, 'Y': Y, 'Z': Z}
    return regions, lines
