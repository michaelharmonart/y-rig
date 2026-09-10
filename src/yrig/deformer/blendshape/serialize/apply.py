from __future__ import annotations

from collections.abc import Collection

from maya.api.OpenMaya import (
    MPoint,
    MPointArray,
)

from yrig.maya_api.attribute import BlendShapeInputTargetAttribute
from yrig.maya_api.node import BlendShape
from yrig.maya_api.utils import get_plug, set_component_list_indices, set_point_array

from ..core import get_name_to_target_index_map
from .data import (
    BlendShapeData,
    BlendShapeTargetDirectory,
    BlendShapeTargetGroupData,
    BlendShapeTargetItemData,
    get_target_group_indices_map,
)
from .directory import prune_blendshape_directory_dict, resolve_needed_group_indices


def get_name_to_directory_index_map(blendshape: BlendShape) -> dict[str, int]:
    name_to_index: dict[str, int] = {}
    for index in blendshape.target_directory.get_indices():
        if index == 0:
            continue
        name_to_index[blendshape.target_directory[index].dirctory_name.get()] = index
    return name_to_index


def _resolve_directory_index_map(
    blendshape: BlendShape,
    directory_data: dict[int, BlendShapeTargetDirectory],
    use_existing: bool = True,
) -> dict[int, int]:
    """Map data-file directory indices -> actual directory plug indices on
    `blendshape`, reusing directories that already exist by name and
    assigning fresh indices for any that don't."""
    name_to_index = get_name_to_directory_index_map(blendshape)
    index_map: dict[int, int] = {0: 0}  # root always maps to root
    next_index = blendshape.target_directory.next_available_index()
    for data_index, directory in directory_data.items():
        if data_index == 0:
            continue
        if directory.name in name_to_index:
            index_map[data_index] = name_to_index[directory.name]
        else:
            index_map[data_index] = next_index
            next_index += 1
    return index_map


def _remap_child_index(
    value: int, directory_index_map: dict[int, int], group_index_map: dict[int, int]
) -> int:
    if value >= 0:
        return group_index_map[value]
    original_directory_index = -value
    return -directory_index_map[original_directory_index]


def apply_blendshape_directory_data(
    blendshape: BlendShape,
    directory_data: dict[int, BlendShapeTargetDirectory],
    group_index_map: dict[int, int],
    parent_directory: str | None,
) -> None:
    directory_index_map = _resolve_directory_index_map(
        blendshape, directory_data, use_existing=parent_directory is None
    )

    for data_index, directory in directory_data.items():
        new_index = directory_index_map[data_index]
        directory_attr = blendshape.target_directory[new_index]
        if data_index != 0:
            directory_attr.dirctory_name.set(directory.name)
            directory_attr.parent_index.set(directory_index_map[directory.parent_index])
        directory_attr.child_indices.set(
            [
                _remap_child_index(child, directory_index_map, group_index_map)
                for child in directory.child_indices
            ]
        )


def _resolve_target_group_index_for_apply(
    blendshape: BlendShape,
    input_target: BlendShapeInputTargetAttribute,
    name_to_target_index_map: dict[str, int],
    target_group_name: str,
    overwrite_existing: bool = True,
) -> int:
    if target_group_name in name_to_target_index_map:
        if overwrite_existing:
            raise RuntimeError(
                f"Target name {target_group_name} already exists on {blendshape} and `overwrite_existing` is not set to True."
            )
        else:
            return name_to_target_index_map[target_group_name]
    return input_target.input_target_group.next_available_index()


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
    target_index: int,
    group_index: int,
) -> None:
    if blendshape.weight[group_index].get_alias() != data.name:
        blendshape.weight[group_index].set_alias(data.name)
    blendshape.weight[group_index].set(0)
    apply_blendshape_target_items_dict(
        blendshape, data.items, target_index=target_index, group_index=group_index
    )


def apply_blendshape_data(
    blendshape: str | BlendShape,
    data: BlendShapeData,
    directories: Collection[str] | None = None,
    targets: Collection[str] | None = None,
    parent_directory: str | None = None,
) -> None:
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

    blendshape_name_to_target_index_map = get_name_to_target_index_map(str(blendshape_node))
    group_index_map: dict[int, int] = {}
    for target_index, target in data.targets.items():
        for group_index, group in target.groups.items():
            if needed_groups_indices is None or group_index in needed_groups_indices:
                apply_index = _resolve_target_group_index_for_apply(
                    blendshape=blendshape_node,
                    input_target=blendshape_node.input_target[target_index],
                    name_to_target_index_map=blendshape_name_to_target_index_map,
                    target_group_name=group.name,
                )
                apply_blendshape_target_group_data(
                    blendshape_node, group, target_index=target_index, group_index=apply_index
                )
                group_index_map[group_index] = apply_index

    if data.directory:
        directory_data = (
            data.directory
            if needed_groups_indices is None
            else prune_blendshape_directory_dict(
                data.directory,
                directories_to_keep=directories or set(),
                group_indices_to_keep=needed_groups_indices,
            )
        )
        apply_blendshape_directory_data(
            blendshape_node, directory_data, group_index_map, parent_directory
        )
