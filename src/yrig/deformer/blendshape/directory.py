from collections.abc import Generator
from contextlib import contextmanager

from yrig.maya_api.node import BlendShapeNode, ShapeEditorManagerNode


@contextmanager
def preserve_shape_editor_root_directory(
    manager: ShapeEditorManagerNode,
) -> Generator[None, None, None]:
    """
    When initializing a blendShape Maya adds it to the root directory.
    This screws up our manual handling of directories on import.

    This is an easy way to guarantee that Maya won't screw up directory data.
    """
    snapshot = manager.blend_shape_directory[0].child_indices.get()
    try:
        yield
    finally:
        manager.blend_shape_directory[0].child_indices.set(snapshot)


def find_child_directory_index(
    manager: ShapeEditorManagerNode,
    parent_directory_index: int,
    name: str,
) -> int | None:
    """Return the index of the direct child directory called ``name``, or None."""
    child_indices = manager.blend_shape_directory[parent_directory_index].child_indices.get()
    for child_index in child_indices:
        if child_index >= 0:  # positive entries are non directory children
            continue
        directory_index = -child_index
        directory = manager.blend_shape_directory[directory_index]
        if directory.directory_name.get() == name:
            return directory_index
    return None


def add_blendshape_to_directory(
    manager: ShapeEditorManagerNode, blendshape: BlendShapeNode, parent_directory_index: int
) -> int:
    child_index = blendshape.mid_layer_id.get()
    blendshape.mid_layer_parent.set(parent_directory_index)
    parent_directory = manager.blend_shape_directory[parent_directory_index]
    original_child_indices = parent_directory.child_indices.get()
    parent_directory.child_indices.set(value=[*original_child_indices, child_index])

    return child_index


def add_shape_editor_directory(
    manager: ShapeEditorManagerNode, name: str, parent_directory_index: int
) -> int:
    directory_index = manager.blend_shape_directory.next_available_index()
    directory = manager.blend_shape_directory[directory_index]
    directory.parent_index.set(parent_directory_index)
    directory.directory_name.set(name)
    parent_directory = manager.blend_shape_directory[parent_directory_index]
    original_child_indices = parent_directory.child_indices.get()
    parent_directory.child_indices.set([*original_child_indices, -directory_index])
    return directory_index
