"""
Skinning utilities for Maya meshes.

Provides tools for querying and manipulating skinCluster weights, splitting
weights across joints using spline-based falloff, ngSkinTools2 integration,
and debug visualization of per-vertex influences.
"""

from . import apply, core, export, ng, serialize, split, visualize
from .apply import (
    skin_and_apply_ng_weights,
    skin_and_apply_weights,
    skin_and_apply_weights_from_directories,
)
from .core import skin_geometry, transfer_skin_weights
from .export import (
    batch_export_skin_weights,
    export_skin_weights_for_selected,
    export_skin_weights_for_shape,
)
from .serialize import export_skin_weights, import_skin_weights

__all__ = [
    "apply",
    "batch_export_skin_weights",
    "core",
    "export",
    "export_skin_weights",
    "export_skin_weights_for_selected",
    "export_skin_weights_for_shape",
    "import_skin_weights",
    "ng",
    "serialize",
    "skin_and_apply_ng_weights",
    "skin_and_apply_weights",
    "skin_and_apply_weights_from_directories",
    "skin_geometry",
    "split",
    "transfer_skin_weights",
    "visualize",
]
