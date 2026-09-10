"""
Serialize and restore Maya blendShape target data.

Notes for future adventures:
The names ``target``, ``group``, and ``item`` refer to Maya's internal
attribute hierarchy.

In particular:

* ``target`` refers to an input geometry slot on the blendShape..

* ``group`` refers to a blendShape weight/alias within that target slot.
  Despite being called a "group" by Maya, this is effectively the named
  blendShape target that an artist sees (for example, ``smile``).

* ``item`` These are the final leaf structures that hold the data to represent
the base target and in-between targets. The base target is stored at index 6000.
This is so that even with only a sparse array, you can write inbetween deltas for
weights from -5 to essentially infinity with 1% increments in precision.
5000 = -1 6000 = 1, 6500 = 1.5 7000 = 2 etc.

Overengineered and weird? Yep :)

The dataclass names intentionally mirror this Maya hierarchy so that the
serialized structure corresponds directly to the underlying attributes.
"""

from . import apply, data, directory, get, io
from .get import get_blendshape_data
from .io import export_blendshape, import_blendshape

__all__ = [
    "apply",
    "data",
    "directory",
    "export_blendshape",
    "get",
    "get_blendshape_data",
    "import_blendshape",
    "io",
]
