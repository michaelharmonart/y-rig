from __future__ import annotations

from collections import deque
from collections.abc import Collection
from dataclasses import replace

from maya.api.OpenMaya import (
    MPoint,
    MPointArray,
)

from yrig.maya_api.node import BlendShape
from yrig.maya_api.utils import get_plug, set_component_list_indices, set_point_array

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


def add_target_group(
    blendshape: BlendShape,
    data: BlendShapeTargetGroupData,
    input_target_index: int,
    parent_directory_index: int = 0,
) -> int:
    """Add target group to blendshape and apply data. Returns the index at which the target was added."""
    target_group_index = blendshape.weight.next_available_index()
    apply_blendshape_target_group_data(blendshape, data, input_target_index, target_group_index)
    child_indices = blendshape.target_directory[parent_directory_index].child_indices.get()
    new_child_indices = child_indices + [target_group_index]
    blendshape.target_directory[parent_directory_index].child_indices.set(new_child_indices)
    return target_group_index


def add_target_directory(
    blendshape: BlendShape, data: BlendShapeTargetDirectory, parent_directory_index: int
) -> int:
    """Add target directory to blendshape and apply data. Returns the index at which the target directory was added."""
    target_directory_index = blendshape.target_directory.next_available_index()
    blendshape.target_directory[target_directory_index].directory_name.set(data.name)
    blendshape.target_directory[target_directory_index].parent_index.set(parent_directory_index)
    parent_directory_child_indices = blendshape.target_directory[
        parent_directory_index
    ].child_indices.get()

    new_child_indices = parent_directory_child_indices + [-target_directory_index]
    blendshape.target_directory[parent_directory_index].child_indices.set(new_child_indices)
    return target_directory_index


def get_existing_child_directories(
    blendshape: BlendShape, target_directory_index: int
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
    blendshape: BlendShape,
    data: BlendShapeData,
    target_directory_index: int,
    parent_directory_index: int = 0,
    target_groups_to_skip: Collection[str] | None = None,
    target_directories_indices_to_skip: Collection[int] | None = None,
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
            child_directory = data.directory[source_directory_index]
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
            )
        # Indices >= 0 are target groups
        else:
            for input_target_index, input_target in data.targets.items():
                if child_index in input_target.groups:
                    target_group_data = input_target.groups[child_index]
                    if (
                        target_groups_to_skip is not None
                        and target_group_data.name in target_groups_to_skip
                    ):
                        continue
                    add_target_group(
                        blendshape,
                        data=target_group_data,
                        input_target_index=input_target_index,
                        parent_directory_index=parent_directory_index,
                    )


def apply_blendshape_target_item_data(
    blendshape: BlendShape,
    data: BlendShapeTargetItemData,
    target_index: int,
    group_index: int,
    item_index: int,
) -> None:
    item_attr = (
        blendshape.input_target[target_index]
        .input_target_group[group_index]
        .input_target_item[item_index]
    )
    component_plug = get_plug(str(item_attr.input_components_target))
    points_plug = get_plug(str(item_attr.input_points_target))

    point_array: MPointArray = MPointArray()
    point_array.setLength(len(data.points))
    for index, point in enumerate(data.points):
        point_array[index] = MPoint(*point)

    set_component_list_indices(component_plug, data.components)
    set_point_array(points_plug, point_array)


def apply_blendshape_target_items_dict(
    blendshape: BlendShape,
    data: dict[int, BlendShapeTargetItemData],
    target_index: int,
    group_index: int,
) -> None:
    for index, target_item in data.items():
        apply_blendshape_target_item_data(
            blendshape,
            data=target_item,
            target_index=target_index,
            group_index=group_index,
            item_index=index,
        )
        blendshape.input_target[target_index].input_target_group[group_index]


def apply_blendshape_target_group_data(
    blendshape: BlendShape,
    data: BlendShapeTargetGroupData,
    input_target_index: int,
    target_group_index: int,
) -> None:
    original_root_child_indices = blendshape.target_directory[0].child_indices.get()
    if blendshape.weight[target_group_index].get_alias() != data.name:
        blendshape.weight[target_group_index].set_alias(data.name)
    blendshape.weight[target_group_index].set(0)
    apply_blendshape_target_items_dict(
        blendshape, data.items, target_index=input_target_index, group_index=target_group_index
    )
    blendshape.target_directory[0].child_indices.set(original_root_child_indices)


def get_parent_directory_index_from_name(
    blendshape: BlendShape, parent_directory: str
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


def apply_blendshape_data(
    blendshape: str | BlendShape,
    data: BlendShapeData,
    directories: Collection[str] | None = None,
    targets: Collection[str] | None = None,
    parent_directory: str | None = None,
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
        blendshape if isinstance(blendshape, BlendShape) else BlendShape.from_existing(blendshape)
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
        parent_directory_index = get_parent_directory_index_from_name(
            blendshape_node, parent_directory
        )
        if parent_directory_index is None:
            raise RuntimeError(
                f"The specified parent directory {parent_directory} couldn't be found on {blendshape_node}"
            )

        # Apply reparented targets first as they're easy.
        if targets is not None:
            data_target_to_index_map = get_target_group_indices_map(data)
            for target_name in targets:
                if target_name not in data_target_to_index_map:
                    raise RuntimeError(
                        f"The target {target_name} couldn't be found in the data to be applied."
                    )
                target_group_index = data_target_to_index_map[target_name]
                for input_target_index, input_target in data.targets.items():
                    if target_group_index in input_target.groups:
                        target_group_data = input_target.groups[target_group_index]
                        add_target_group(
                            blendshape_node,
                            data=target_group_data,
                            input_target_index=input_target_index,
                            parent_directory_index=parent_directory_index,
                        )
                        already_added_targets.add(target_name)

    if directories is not None:
        directory_indices_map = get_directory_indices(directory_data, directories)
        directory_indices_set = set(directory_indices_map.values())
        existing_child_directories_map = get_existing_child_directories(blendshape_node, 0)
        for directory_index in directory_indices_map.values():
            directory_name = pruned_data.directory[directory_index].name
            maya_directory_index = existing_child_directories_map.get(directory_name)
            if maya_directory_index is None:
                maya_directory_index = add_target_directory(
                    blendshape_node,
                    pruned_data.directory[directory_index],
                    parent_directory_index=parent_directory_index,
                )
            apply_directory_children(
                blendshape_node,
                pruned_data,
                target_directory_index=directory_index,
                parent_directory_index=maya_directory_index,
                target_groups_to_skip=already_added_targets,
                target_directories_indices_to_skip=directory_indices_set,
            )
    else:
        apply_directory_children(
            blendshape_node,
            pruned_data,
            target_directory_index=0,
            parent_directory_index=parent_directory_index,
            target_groups_to_skip=already_added_targets,
        )
