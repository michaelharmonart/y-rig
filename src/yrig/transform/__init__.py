"""Transform utilities for Maya rigs.

Provides helpers for querying and manipulating Maya transform nodes,
including world/local matrix operations, matrix-based constraints,
and common transform tasks such as matching, zeroing, and reparenting.
"""

from . import constraint, matrix, quat, utils
from .constraint import matrix_constraint
from .matrix import (
    get_local_matrix,
    get_parent_inverse_matrix,
    get_parent_matrix,
    is_identity_matrix,
    mmatrix_to_list,
    set_world_matrix,
)
from .utils import (
    create_transform,
    get_shapes,
    match_location,
    match_transform,
    zero_rotate_axis,
)

__all__ = [
    "constraint",
    # Utils
    "create_transform",
    # Matrix
    "get_local_matrix",
    "get_parent_inverse_matrix",
    "get_parent_matrix",
    "get_shapes",
    "is_identity_matrix",
    "match_location",
    "match_transform",
    "matrix",
    "matrix_constraint",
    "mmatrix_to_list",
    "quat",
    "set_world_matrix",
    "utils",
    "zero_rotate_axis",
]
