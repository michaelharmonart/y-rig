"""
Matrix spline construction and evaluation.
"""

from . import build, core, pin
from .core import (
    MatrixSpline,
    bound_curve_from_matrix_spline,
    closest_parameter_on_matrix_spline,
)
from .pin import pin_to_matrix_spline

__all__ = [
    "MatrixSpline",
    "bound_curve_from_matrix_spline",
    "build",
    "closest_parameter_on_matrix_spline",
    "core",
    "pin",
    "pin_to_matrix_spline",
]
