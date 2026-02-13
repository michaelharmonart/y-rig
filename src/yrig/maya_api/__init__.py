"""
Maya API abstraction layer.

Provides Pythonic wrappers around Maya's dependency-graph nodes and their
attributes so that node creation, attribute access, and connection wiring
can be expressed with clean, type-safe Python instead of raw ``cmds`` calls.

Submodules:
    attribute: Typed attribute descriptors (scalar, matrix, vector, etc.).
    node: Convenience classes for common Maya DG/DAG node types.
"""

from . import attribute as attribute
from . import node as node
