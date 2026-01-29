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
from .maya_query import get_cv_weights, get_cvs, get_knots, maya_to_standard_knots

__all__ = [
    # Math
    "generate_knots",
    "is_periodic_knot_vector",
    "point_on_spline_weights",
    "get_weights_along_spline",
    "tangent_on_spline_weights",
    "get_point_on_spline",
    "get_tangent_on_spline",
    "resample",
    # Maya Query
    "get_knots",
    "maya_to_standard_knots",
    "get_cvs",
    "get_cv_weights",
]
