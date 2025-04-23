import torch
import numpy as np


def get_projection_matrix_and_centroid(data: torch.Tensor):
    """
    Compute a 2D projection matrix (principal components) and centroid for n-D data.
    Args:
        data: Tensor of shape (N, D) or any shape where last dim is features.
    Returns:
        T: Tensor of shape (2, D) mapping D->2
        x0: Tensor of shape (D,) the data centroid
    """
    pts = data.reshape(-1, data.shape[-1])  # (N, D)
    x0 = pts.mean(dim=0)
    pts_centered = pts - x0
    # SVD for PCA
    U, S, Vt = torch.linalg.svd(pts_centered, full_matrices=False)
    # Vt: (D, D), take first two principal directions
    T = Vt[:2]  # (2, D)
    return T, x0


def get_mgrid(domain, grid_size=100):
    """
    Create a meshgrid inside a 2D polygon domain by projecting to 2D
    and back-projecting to original space.
    Args:
        domain: Tensor (M, D) polygon vertices in D dims
        grid_size: int resolution per axis
    Returns:
        grid_points: Tensor (grid_size**2, D)
        grid_xy: tuple of 2D arrays (X, Y)
    """
    from matplotlib.path import Path
    # Project domain to 2D
    T, x0 = get_projection_matrix_and_centroid(domain)
    domain_2d = (T @ (domain - x0).T).T.numpy()
    # bounding box
    x_min, y_min = domain_2d.min(axis=0)
    x_max, y_max = domain_2d.max(axis=0)
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xs, ys)
    pts2d = np.stack([X.ravel(), Y.ravel()], axis=-1)
    # back-project
    P_pinv = torch.linalg.pinv(T)
    pts_nd = (torch.from_numpy(pts2d) @ P_pinv.T) + x0
    # mask outside polygon
    mask = Path(domain_2d).contains_points(pts2d)
    grid_points = pts_nd[mask]
    return grid_points, (X, Y, mask.reshape(X.shape))
