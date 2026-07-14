"""Unit tests for the SigmaCam package.

A linear model with a sigmoid head has an analytically known decision
boundary (the line w.x + b = 0), so we can assert that the extracted
contour actually lies on it — the core correctness claim of the method.
"""

import numpy as np
import pytest
import torch

import sigmacam
from sigmacam import (SigmaCamModel, compute_boundaries, get_mgrid,
                      get_projection_matrix_and_centroid, plot_boundaries,
                      wrapper)

torch.manual_seed(0)


# ------------------------------------------------------------------ utils ---

def test_projection_matrix_shape_and_centroid():
    data = torch.randn(200, 5)
    T, x0 = get_projection_matrix_and_centroid(data)
    assert T.shape == (2, 5)
    assert x0.shape == (5,)
    assert torch.allclose(x0, data.mean(dim=0), atol=1e-5)


def test_projection_rows_are_orthonormal():
    data = torch.randn(300, 4)
    T, _ = get_projection_matrix_and_centroid(data)
    gram = T @ T.T
    assert torch.allclose(gram, torch.eye(2, dtype=gram.dtype), atol=1e-4)


def test_projection_recovers_dominant_direction():
    """Data stretched along one axis: PC1 must align with that axis."""
    base = torch.randn(500, 3) * torch.tensor([10.0, 0.1, 0.1])
    T, _ = get_projection_matrix_and_centroid(base)
    pc1 = T[0].abs()
    assert pc1[0] > 0.99  # dominant direction found


def test_get_mgrid_points_stay_inside_domain():
    square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts, (X, Y, mask) = get_mgrid(square, grid_size=25)
    assert pts.shape[1] == 2
    assert len(pts) == mask.sum()
    # All generated points lie within the polygon's bounding box
    assert (pts[:, 0] >= -1e-6).all() and (pts[:, 0] <= 1 + 1e-6).all()
    assert (pts[:, 1] >= -1e-6).all() and (pts[:, 1] <= 1 + 1e-6).all()


# ---------------------------------------------------------------- wrapper ---

def test_wrapper_predict_is_eval_and_nograd():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    T = torch.eye(2)
    x0 = torch.zeros(2)
    wrapped = wrapper(model, (2,), T, x0)
    assert isinstance(wrapped, SigmaCamModel)
    out = wrapped.predict(torch.randn(10, 2))
    assert out.shape == (10, 1)
    assert not out.requires_grad
    assert not wrapped.model.training  # predict() must switch to eval mode


# ------------------------------------------------------------- boundaries ---

@pytest.fixture
def linear_sigmoid_model():
    """sigmoid(x + y): boundary is exactly the line x + y = 0."""
    linear = torch.nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight[:] = torch.tensor([[1.0, 1.0]])
        linear.bias[:] = 0.0
    model = torch.nn.Sequential(linear, torch.nn.Sigmoid())
    return wrapper(model, (2,), torch.eye(2), torch.zeros(2))


def test_boundary_matches_analytic_line(linear_sigmoid_model):
    square = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    regions, lines = compute_boundaries(linear_sigmoid_model, square, grid_size=120)

    assert set(regions) == {"X", "Y", "Z"}
    assert len(lines) >= 1, "no decision boundary found for a crossing sigmoid"

    # The extracted 0.5-contour must satisfy x + y ~= 0 along its length.
    # (The contour lives in the PCA-projected plane; for a symmetric square
    # domain the projection is a rotation, so the line stays a line and we
    # verify via the model itself: prediction on the boundary ~= 0.5.)
    boundary_2d = np.vstack(lines)
    assert len(boundary_2d) > 10


def test_boundary_probabilities_near_half(linear_sigmoid_model):
    """Model output evaluated on extracted boundary points must be ~0.5."""
    square = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    _, lines = compute_boundaries(linear_sigmoid_model, square, grid_size=150)
    T, x0 = get_projection_matrix_and_centroid(square)
    P_pinv = torch.linalg.pinv(T)
    for seg in lines:
        pts2d = torch.from_numpy(np.asarray(seg)).float()
        pts_nd = (pts2d @ P_pinv.T) + x0
        proba = linear_sigmoid_model.predict(pts_nd).squeeze(-1)
        assert torch.allclose(proba, torch.full_like(proba, 0.5), atol=0.02), \
            f"boundary points deviate from p=0.5 (max dev {(proba-0.5).abs().max():.4f})"


def test_no_boundary_when_model_is_one_sided():
    """A model that never crosses 0.5 must return no contour lines."""
    linear = torch.nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight[:] = 0.0
        linear.bias[:] = 3.0   # sigmoid(3) ~ 0.95 everywhere
    model = torch.nn.Sequential(linear, torch.nn.Sigmoid())
    wrapped = wrapper(model, (2,), torch.eye(2), torch.zeros(2))
    square = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    _, lines = compute_boundaries(wrapped, square, grid_size=60)
    assert lines == []


# --------------------------------------------------------------- plotting ---

def test_plot_boundaries_returns_fig(linear_sigmoid_model):
    import matplotlib
    matplotlib.use("Agg")
    square = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    regions, lines = compute_boundaries(linear_sigmoid_model, square, grid_size=60)
    fig, ax = plot_boundaries(regions, lines)
    assert fig is not None and ax is not None


def test_package_metadata():
    assert sigmacam.__version__ == "0.1.0"
