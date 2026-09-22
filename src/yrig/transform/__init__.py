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
    get_position,
    get_shapes,
    get_transform,
    match_location,
    match_transform,
    partial_path_name,
    set_position,
    zero_rotate_axis,
    zero_transform,
)

__all__ = [
    "constraint",
    "create_transform",
    "get_local_matrix",
    "get_parent_inverse_matrix",
    "get_parent_matrix",
    "get_position",
    "get_shapes",
    "get_transform",
    "is_identity_matrix",
    "match_location",
    "match_transform",
    "matrix",
    "matrix_constraint",
    "mmatrix_to_list",
    "partial_path_name",
    "quat",
    "set_position",
    "set_world_matrix",
    "utils",
    "zero_rotate_axis",
    "zero_transform",
]
