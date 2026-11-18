"""
This package is responsible only for coordinating rig builds through external
APIs and integrations, such as mGear, NXT, and other build-time systems.

General-purpose rigging functionality, higher-level helpers, and reusable Maya
operations belong in `yrig`, not in this package.
"""

from . import metadata, mgear_api, nxt_api, setup
from .core import build_rig, open_rig_in_editor
from .scope import BuildScope

__all__ = [
    "BuildScope",
    "build_rig",
    "metadata",
    "mgear_api",
    "nxt_api",
    "open_rig_in_editor",
    "setup",
]
