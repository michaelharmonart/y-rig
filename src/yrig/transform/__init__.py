"""Transform utilities for Maya rigs.

Provides helpers for querying and manipulating Maya transform nodes,
including world/local matrix operations, matrix-based constraints,
and common transform tasks such as matching, zeroing, and reparenting.
"""

from . import matrix as matrix
from . import utils as utils
from .matrix import (
    get_local_matrix,
    get_parent_inverse_matrix,
    get_parent_matrix,
    is_identity_matrix,
    matrix_constraint,
    mmatrix_to_list,
    set_world_matrix,
)
from .utils import get_shapes, match_location, match_transform, zero_rotate_axis

__all__ = [
    # Matrix
    "get_local_matrix",
    "get_parent_inverse_matrix",
    "get_parent_matrix",
    "is_identity_matrix",
    "matrix_constraint",
    "mmatrix_to_list",
    "set_world_matrix",
    # Utils
    "get_shapes",
    "match_location",
    "match_transform",
    "zero_rotate_axis",
]
