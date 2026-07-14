"""SigmaCam: exact decision-boundary extraction for DNNs with smooth activations.

Published at IEEE IJCNN 2025 (first author: Nikkat Afrin).
"""

from .boundaries import compute_boundaries
from .plotting import plot_boundaries
from .utils import get_mgrid, get_projection_matrix_and_centroid
from .wrapper import SigmaCamModel, wrapper

__version__ = "0.1.0"

__all__ = [
    "SigmaCamModel",
    "compute_boundaries",
    "get_mgrid",
    "get_projection_matrix_and_centroid",
    "plot_boundaries",
    "wrapper",
    "__version__",
]
