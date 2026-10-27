from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

from maya.api.OpenMaya import (
    MDagPath,
    MFnComponent,
    MFnComponentListData,
    MFnDoubleIndexedComponent,
    MObject,
    MPlug,
    MPoint,
    MPointArray,
)

from yrig.maya_api.node import BlendShapeNode
from yrig.maya_api.utils import get_dag_path, get_plug, set_point_array
from yrig.shape import get_component_mfn_and_type_of_shape

from .data import (
    BlendShapeData,
    BlendShapeTargetDirectory,
    BlendShapeTargetGroupData,
    BlendShapeTargetItemData,
    get_target_group_indices_map,
)
from .directory import (
    get_directory_indices,
    prune_blendshape_directory_dict,
    resolve_needed_group_indices,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Generator

    from yrig.maya_api.attribute import (
        BlendShapeInputTargetGroupAttribute,
        BlendShapeInputTargetItemAttribute,
    )

from enum import StrEnum

log = logging.getLogger(__name__)


class ExistingPolicy(StrEnum):
    UPDATE = "update"
    SKIP = "skip"
    DUPLICATE = "duplicate"


@contextmanager
def preserve_all_target_directories(blendshape: BlendShapeNode) -> Generator[None, None, None]:
    """
    When initializing a new inputTargetGroup plug Maya just decides a directory to add it to.
    This screws up our manual handling of directories on import.

    This might be overkill, but since I don't have the Maya source code on hand
    this is the only way to guarantee that Maya won't screw up directory data.
    """
    directory_indices = blendshape.target_directory.get_indices()
    snapshot = {
        index: blendshape.target_directory[index].child_indices.get() for index in directory_indices
    }
    try:
        yield
    finally:
        for index, child_indices in snapshot.items():
            blendshape.target_directory[index].child_indices.set(child_indices)


def set_component_list_indices(
    plug: MPlug,
    mfn_component_type: type[MFnComponent],
    component_type: int,
    indices: list[int | tuple[int, int]],
) -> None:
    fn_data: MFnComponentListData = MFnComponentListData()
    data_mob: MObject = fn_data.create()
    fn_comp = mfn_component_type()
    comp_mob: MObject = fn_comp.create(component_type)
    fn_comp.addElements(indices)
    fn_data.add(comp_mob)
    plug.setMObject(data_mob)


def find_child_blendshape_target(
    blendshape: BlendShapeNode,
    parent_directory_index: int,
    name: str,
) -> int | None:
    """Return the index of a target directly inside the directory, or None."""
    child_indices = blendshape.target_directory[parent_directory_index].child_indices.get()

    for child_index in child_indices:
        if child_index < 0:
            continue

        if blendshape.weight[child_index].get_alias() == name:
            return child_index

    return None


def merge_target_group(
    blendshape: BlendShapeNode,
    data: BlendShapeTargetGroupData,
    input_target_index: int,
    parent_directory_index: int,
    existing_policy: ExistingPolicy = ExistingPolicy.UPDATE,
) -> int | None:
    """Add a target group, or update an existing target with the same name."""
    if existing_policy != ExistingPolicy.DUPLICATE:
        existing_index = find_child_blendshape_target(blendshape, parent_directory_index, data.name)
        if existing_index is not None:
            if existing_policy == ExistingPolicy.SKIP:
                return None

            input_geometry_attr = blendshape.input[input_target_index].input_geometry.get_input()
            if input_geometry_attr is None:
                raise RuntimeError(f"No input geometry connected to target {input_target_index}.")

            input_geometry = get_dag_path(str(input_geometry_attr).split(".", 1)[0])
            target_group = blendshape.input_target[input_target_index].input_target_group[
                existing_index
            ]
            with preserve_all_target_directories(blendshape):
                target_group.input_target_item.clear()
                apply_blendshape_target_group_data(
                    blendshape,
                    data,
                    input_geometry,
                    input_target_index,
                    existing_index,
                )

            return existing_index

    return add_target_group(
        blendshape,
        data,
        input_target_index,
        parent_directory_index,
    )


def add_target_group(
    blendshape: BlendShapeNode,
    data: BlendShapeTargetGroupData,
    input_target_index: int,
    parent_directory_index: int = 0,
) -> int:
    """Add target group to blendshape and apply data. Returns the index at which the target was added."""
    target_group_index = blendshape.weight.next_available_index()

    parent_directory = blendshape.target_directory[parent_directory_index]
    original_child_indices = parent_directory.child_indices.get()
    input_geometry_attr = blendshape.input[input_target_index].input_geometry.get_input()
    if input_geometry_attr is None:
        raise RuntimeError(f"{input_geometry_attr} had no input geometry.")
    input_geometry_node = str(input_geometry_attr).split(".", 1)[0]
    input_geometry = get_dag_path(input_geometry_node)
    with preserve_all_target_directories(blendshape):
        apply_blendshape_target_group_data(
            blendshape, data, input_geometry, input_target_index, target_group_index
        )
    parent_directory.child_indices.set([*original_child_indices, target_group_index])
    return target_group_index


def add_target_directory(
    blendshape: BlendShapeNode, data: BlendShapeTargetDirectory, parent_directory_index: int
) -> int:
    """Add target directory to blendshape and apply data. Returns the index at which the target directory was added."""
    target_directory_index = blendshape.target_directory.next_available_index()
    target_directory = blendshape.target_directory[target_directory_index]

    target_directory.directory_name.set(data.name)
    target_directory.parent_index.set(parent_directory_index)
    parent_directory_child_indices = blendshape.target_directory[
        parent_directory_index
    ].child_indices.get()

    new_child_indices = [*parent_directory_child_indices, -target_directory_index]
    blendshape.target_directory[parent_directory_index].child_indices.set(new_child_indices)
    return target_directory_index


def get_existing_child_directories(
    blendshape: BlendShapeNode, target_directory_index: int
) -> dict[str, int]:
    directory = blendshape.target_directory[target_directory_index]
    # Get existing directories
    existing_directory_map: dict[str, int] = {}
    for child_index in directory.child_indices.get():
        # Indices < 0 are directories
        if child_index < 0:
            directory_index = -child_index
            directory_name = blendshape.target_directory[directory_index].directory_name.get()
            existing_directory_map[directory_name] = directory_index
    return existing_directory_map


def apply_directory_children(
    blendshape: BlendShapeNode,
    data: BlendShapeData,
    target_directory_index: int,
    parent_directory_index: int = 0,
    target_groups_to_skip: Collection[str] | None = None,
    target_directories_indices_to_skip: Collection[int] | None = None,
    existing_policy: ExistingPolicy = ExistingPolicy.UPDATE,
) -> None:
    target_directory_to_apply = data.directory[target_directory_index]
    existing_directory_map = get_existing_child_directories(blendshape, parent_directory_index)
    for child_index in target_directory_to_apply.child_indices:
        # Indices < 0 are directories
        if child_index < 0:
            source_directory_index = -child_index
            if (
                target_directories_indices_to_skip is not None
                and source_directory_index in target_directories_indices_to_skip
            ):
                continue

            child_directory = data.directory.get(source_directory_index)
            if child_directory is None:
                log.warning(
                    f"No directory with index {child_index} found in data. It's likely malformed due to Maya not removing deleted directory from the parent directory."
                )
                continue
            maya_directory_index = existing_directory_map.get(child_directory.name)
            if maya_directory_index is None:
                maya_directory_index = add_target_directory(
                    blendshape,
                    child_directory,
                    parent_directory_index=parent_directory_index,
                )
            apply_directory_children(
                blendshape,
                data,
                target_directory_index=source_directory_index,
                parent_directory_index=maya_directory_index,
                target_groups_to_skip=target_groups_to_skip,
                target_directories_indices_to_skip=target_directories_indices_to_skip,
                existing_policy=existing_policy,
            )
        # Indices >= 0 are target groups
        else:
            for input_target_index, input_target in data.targets.items():
                if child_index in input_target.groups:
                    target_group_data = input_target.groups.get(child_index)
                    if target_group_data is None:
                        log.warning(
                            f"No target group with index {child_index} found in data. It's likely malformed due to Maya not removing deleted targets from the parent directory."
                        )
                        continue
                    if (
                        target_groups_to_skip is not None
                        and target_group_data.name in target_groups_to_skip
                    ):
                        continue
                    merge_target_group(
                        blendshape,
                        data=target_group_data,
                        input_target_index=input_target_index,
                        parent_directory_index=parent_directory_index,
                        existing_policy=existing_policy,
                    )


def apply_blendshape_target_item_data(
    data: BlendShapeTargetItemData,
    input_geometry: MDagPath,
    target_item: BlendShapeInputTargetItemAttribute,
) -> None:

    points_plug = get_plug(target_item.input_points_target)
    points_to_apply = data.points or [(0, 0, 0)]  # Maya crashes if there are NO points
    point_array: MPointArray = MPointArray()
    point_array.setLength(len(points_to_apply))
    for index, point in enumerate(points_to_apply):
        point_array[index] = MPoint(*point)
    set_point_array(points_plug, point_array)

    component_mfn, component_type = get_component_mfn_and_type_of_shape(input_geometry)
    component_plug = get_plug(target_item.input_components_target)
    # Maya crashes if there are NO components
    components_to_apply: list[int | tuple[int, int]] = data.components or (
        [(0, 0)] if component_mfn is MFnDoubleIndexedComponent else [0]
    )
    set_component_list_indices(component_plug, component_mfn, component_type, components_to_apply)


def apply_blendshape_target_items_dict(
    data: dict[int, BlendShapeTargetItemData],
    input_geometry: MDagPath,
    target_group: BlendShapeInputTargetGroupAttribute,
) -> None:
    for index, target_item in data.items():
        apply_blendshape_target_item_data(
            data=target_item,
            input_geometry=input_geometry,
            target_item=target_group.input_target_item[index],
        )


def apply_blendshape_target_group_data(
    blendshape: BlendShapeNode,
    data: BlendShapeTargetGroupData,
    input_geometry: MDagPath,
    input_target_index: int,
    target_group_index: int,
) -> None:
    if blendshape.weight[target_group_index].get_alias() != data.name:
        blendshape.weight[target_group_index].set_alias(data.name)
    if blendshape.weight[target_group_index].is_settable():
        blendshape.weight[target_group_index].set(0)
    target_group = blendshape.input_target[input_target_index].input_target_group[
        target_group_index
    ]
    target_group.post_deformers_mode.set(data.mode)
    apply_blendshape_target_items_dict(data.items, input_geometry, target_group)


def get_parent_directory_index_from_name(
    blendshape: BlendShapeNode, parent_directory: str
) -> int | None:
    """
    Breadth first search for the given parent directory by name.
    """
    queue = deque([0])
    while queue:
        current_index = queue.popleft()
        directory = blendshape.target_directory[current_index]
        directory_name = directory.directory_name.get()
        if directory_name == parent_directory:
            return current_index
        queue.extend(-child_index for child_index in directory.child_indices.get())
    return None


def _resolve_parent_directory_index(
    blendshape: BlendShapeNode, parent_directory: str | None
) -> int:
    if not parent_directory:
        return 0
    index = get_parent_directory_index_from_name(blendshape, parent_directory)
    if index is None:
        raise RuntimeError(
            f"The specified parent directory {parent_directory} couldn't be found on {blendshape}"
        )
    return index


def _apply_explicit_targets(
    blendshape: BlendShapeNode,
    data: BlendShapeData,
    target_names: Collection[str],
    target_index_map: dict[str, int],
    parent_directory_index: int,
    existing_policy: ExistingPolicy,
) -> set[str]:
    """Reparent explicitly-named targets directly under parent_directory_index. Returns names applied."""
    applied: set[str] = set()
    for target_name in target_names:
        if target_name not in target_index_map:
            raise RuntimeError(
                f"The target {target_name} couldn't be found in the data to be applied."
            )
        target_group_index = target_index_map[target_name]
        for input_target_index, input_target in data.targets.items():
            if target_group_index in input_target.groups:
                target_group_data = input_target.groups[target_group_index]
                merge_target_group(
                    blendshape,
                    data=target_group_data,
                    input_target_index=input_target_index,
                    parent_directory_index=parent_directory_index,
                    existing_policy=existing_policy,
                )
                applied.add(target_name)
    return applied


def _apply_directories(
    blendshape: BlendShapeNode,
    data: BlendShapeData,
    directories: Collection[str],
    parent_directory_index: int,
    target_groups_to_skip: set[str],
    existing_policy: ExistingPolicy,
) -> None:
    directory_indices_map = get_directory_indices(data.directory, directories)
    directory_indices_set = set(directory_indices_map.values())
    existing_child_directories_map = get_existing_child_directories(
        blendshape, parent_directory_index
    )
    for directory_index in directory_indices_map.values():
        directory_name = data.directory[directory_index].name
        maya_directory_index = existing_child_directories_map.get(directory_name)
        if maya_directory_index is None:
            maya_directory_index = add_target_directory(
                blendshape,
                data.directory[directory_index],
                parent_directory_index=parent_directory_index,
            )
        apply_directory_children(
            blendshape,
            data,
            target_directory_index=directory_index,
            parent_directory_index=maya_directory_index,
            target_groups_to_skip=target_groups_to_skip,
            target_directories_indices_to_skip=directory_indices_set,
            existing_policy=existing_policy,
        )


def apply_blendshape_data(
    blendshape: str | BlendShapeNode,
    data: BlendShapeData,
    *,
    directories: Collection[str] | None = None,
    targets: Collection[str] | None = None,
    parent_directory: str | None = None,
    existing_policy: ExistingPolicy = ExistingPolicy.UPDATE,
) -> None:
    """
    Apply blendShape data to a designated blendshape node.

    Args:
        blendshape: BlendShape node to apply to.
        data: BlendShapeData to use.
        directories: Specify target directories to import.
        targets: Specify target names or indices to export.
        parent_directory: Parent directory for the specified directories and targets,
            or the directory to parent the entire imported structure on if neither are specified.
    """

    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    data_target_to_index_map = get_target_group_indices_map(data)
    data_target_group_indices = (
        {
            data_target_to_index_map[target]
            for target in targets
            if target in data_target_to_index_map
        }
        if targets is not None
        else None
    )
    needed_groups_indices = resolve_needed_group_indices(
        data.directory, directories, data_target_group_indices
    )
    directory_data = (
        data.directory
        if needed_groups_indices is None
        else prune_blendshape_directory_dict(
            data.directory,
            directories_to_keep=directories or set(),
            group_indices_to_keep=needed_groups_indices,
        )
    )
    pruned_data = replace(data, directory=directory_data)

    parent_directory_index = 0
    already_added_targets: set[str] = set()
    if parent_directory:
        parent_directory_index = _resolve_parent_directory_index(blendshape_node, parent_directory)
        # Apply reparented targets first as they're easy.
        if targets is not None:
            already_added_targets = _apply_explicit_targets(
                blendshape_node,
                data,
                targets,
                data_target_to_index_map,
                parent_directory_index,
                existing_policy,
            )

    if directories is not None:
        _apply_directories(
            blendshape_node,
            pruned_data,
            directories,
            parent_directory_index=parent_directory_index,
            target_groups_to_skip=already_added_targets,
            existing_policy=existing_policy,
        )
    else:
        apply_directory_children(
            blendshape_node,
            pruned_data,
            target_directory_index=0,
            parent_directory_index=parent_directory_index,
            target_groups_to_skip=already_added_targets,
            existing_policy=existing_policy,
        )
