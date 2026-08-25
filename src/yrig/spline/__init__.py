"""
Spline rigging and math utilities.

This package provides a collection of functions for working with NURBS splines,
including core mathematical algorithms.
"""

from .math import (
    generate_knots,
    get_point_on_spline,
    get_tangent_on_spline,
    get_weights_along_spline,
    is_periodic_knot_vector,
    point_on_spline_weights,
    resample,
    tangent_on_spline_weights,
)
from .matrix_spline import (
    MatrixSpline,
    closest_parameter_on_matrix_spline,
    pin_to_matrix_spline,
)
from .maya_query import get_cv_weights, get_cvs, get_knots, maya_to_standard_knots

__all__ = [
    # Matrix Spline
    "MatrixSpline",
    "closest_parameter_on_matrix_spline",
    # Math
    "generate_knots",
    "get_cv_weights",
    "get_cvs",
    # Maya Query
    "get_knots",
    "get_point_on_spline",
    "get_tangent_on_spline",
    "get_weights_along_spline",
    "is_periodic_knot_vector",
    "maya_to_standard_knots",
    "pin_to_matrix_spline",
    "point_on_spline_weights",
    "resample",
    "tangent_on_spline_weights",
]
