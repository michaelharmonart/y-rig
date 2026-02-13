"""
Skinning utilities for Maya meshes.

Provides tools for querying and manipulating skinCluster weights, splitting
weights across joints using spline-based falloff, ngSkinTools2 integration,
and debug visualization of per-vertex influences.
"""

from . import core as core
from . import ng as ng
from . import split as split
from . import visualize as visualize
