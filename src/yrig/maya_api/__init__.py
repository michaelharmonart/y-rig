"""
Agnostic wrapper around Maya's Python APIs.

This package provides an abstraction layer over Maya's APIs so that y-rig code
can interact with Maya through nice typesafe Python interfaces rather
than depending directly on schnasty ``maya.cmds`` calls.

The goal is for this package to remain agnostic of higher-level rigging
workflows and external build systems. It is intended to eventually become its
own standalone library that can be used independently of y-rig.

Higher-level rigging helpers and general-purpose workflow logic belong in
:mod:`yrig`.
"""

from . import attribute as attribute
from . import enum as enum
from . import node as node
from . import utils as utils
from . import version as version
from .version import MAYA_API_VERSION as MAYA_API_VERSION
from .version import TARGET_API_VERSION as TARGET_API_VERSION
