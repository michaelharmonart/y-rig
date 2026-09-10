from collections.abc import Collection, Iterable
from dataclasses import replace

from maya.api.OpenMaya import (
    MFnPointArrayData,
    MObject,
    MPoint,
    MPointArray,
)

from yrig.maya_api.attribute import (
    BlendShapeInputTargetAttribute,
    BlendShapeInputTargetGroupAttribute,
    BlendShapeInputTargetItemAttribute,
)
from yrig.maya_api.node import BlendShape
from yrig.maya_api.utils import get_component_indices, get_plug

from ..core import get_target_name_map, resolve_target_index
from .data import (
    BlendShapeData,
    BlendShapeInputData,
    BlendShapeInputTargetData,
    BlendShapeTargetDirectory,
    BlendShapeTargetGroupData,
    BlendShapeTargetItemData,
)


def get_blendshape_target_item_data(
    target_item: BlendShapeInputTargetItemAttribute,
) -> BlendShapeTargetItemData:
    components_plug = get_plug(str(target_item.input_components_target))
    component_ids = get_component_indices(components_plug)

    points_plug = get_plug(str(target_item.input_points_target))
    points_mob: MObject = points_plug.asMObject()
    fn_points: MFnPointArrayData = MFnPointArrayData(points_mob)
    points_array: MPointArray = fn_points.array()
    filtered_components: list[int] = []
    filtered_point_tuples: list[tuple[float, float, float]] = []
    for component_index, point in zip(component_ids, points_array, strict=True):  # type: ignore
        if not point.isEquivalent(MPoint.kOrigin):
            filtered_components.append(component_index)
            filtered_point_tuples.append((point.x, point.y, point.z))
    return BlendShapeTargetItemData(components=filtered_components, points=filtered_point_tuples)


def get_blendshape_target_items_dict(
    target_group: BlendShapeInputTargetGroupAttribute,
) -> dict[int, BlendShapeTargetItemData]:
    items: dict[int, BlendShapeTargetItemData] = {}
    for index in target_group.input_target_item.get_indices():
        item = target_group.input_target_item[index]
        item_data = get_blendshape_target_item_data(item)
        items[index] = item_data
    return items


def get_blendshape_target_groups_dict(
    blendshape: BlendShape,
    target: BlendShapeInputTargetAttribute,
    target_group_indices: Iterable[int] | None = None,
) -> dict[int, BlendShapeTargetGroupData]:
    target_groups: dict[int, BlendShapeTargetGroupData] = {}
    alias_map = get_target_name_map(str(blendshape))
    indices = (
        target_group_indices
        if target_group_indices is not None
        else target.input_target_group.get_indices()
    )
    for index in indices:
        target_group = target.input_target_group[index]
        target_name = alias_map[index]
        target_group_data = BlendShapeTargetGroupData(
            name=target_name, items=get_blendshape_target_items_dict(target_group)
        )
        target_groups[index] = target_group_data

    return target_groups


def get_blendshape_target_dict(
    blendshape: BlendShape, target_group_indices: Iterable[int] | None = None
) -> dict[int, BlendShapeInputTargetData]:
    target_data_dict: dict[int, BlendShapeInputTargetData] = {}
    indices = blendshape.input.get_indices()
    for index in indices:
        target = blendshape.input_target[index]
        group_data = get_blendshape_target_groups_dict(blendshape, target, target_group_indices)
        target_data = BlendShapeInputTargetData(groups=group_data)
        if target_data.groups:
            target_data_dict[index] = target_data
    return target_data_dict


def get_blendshape_input_data_list(blendshape: BlendShape) -> list[BlendShapeInputData]:
    inputs: list[BlendShapeInputData] = []
    indices = blendshape.input.get_indices()
    for index in indices:
        input = blendshape.input[index]
        input_geo = input.input_geometry.get_input()
        original_geo = blendshape.original_geometry[index].get_input()
        if input_geo:
            input_data = BlendShapeInputData(
                str(input_geo),
                original_geometry=str(original_geo) if original_geo else None,
                group_id=input.group_id.get(),
                component_tag_expression=input.component_tag_expression.get(),
            )
            inputs.append(input_data)
    return inputs


def _compute_needed_indices(
    directory_data: dict[int, BlendShapeTargetDirectory],
    directories_to_keep: Collection[str],
    group_indices_to_keep: Collection[int],
) -> set[int]:
    """
    Walk the directory tree and return the set of directory indices
    (negative) and group indices (positive) that must be retained given the
    requested directory names and/or explicit group indices.
    """
    needed_indices: set[int] = set()

    def mark_children(index: int) -> None:
        if index in needed_indices:
            return
        needed_indices.add(index)
        if index < 0:
            directory = directory_data[-index]
            for child in directory.child_indices:
                mark_children(child)

    def mark_needed(index: int) -> bool:
        if index >= 0:
            return index in group_indices_to_keep
        directory = directory_data[-index]
        if directory.name in directories_to_keep:
            mark_children(index)
            return True
        if any(mark_needed(child) for child in directory.child_indices):
            needed_indices.add(index)
            return True
        return False

    for index in directory_data:
        if index != 0:
            mark_needed(-index)

    return needed_indices


def _prune_blendshape_directory_dict(
    directory_data: dict[int, BlendShapeTargetDirectory],
    directories_to_keep: Collection[str],
    group_indices_to_keep: Collection[int],
) -> dict[int, BlendShapeTargetDirectory]:
    needed_indices = _compute_needed_indices(
        directory_data, directories_to_keep, group_indices_to_keep
    )

    pruned_directory_data = {
        index: replace(
            data,
            child_indices=[
                child
                for child in data.child_indices
                if child in needed_indices or child in group_indices_to_keep
            ],
        )
        for index, data in directory_data.items()
        if index == 0 or -index in needed_indices or index in group_indices_to_keep
    }

    return pruned_directory_data


def resolve_needed_group_indices(
    directory_data: dict[int, BlendShapeTargetDirectory],
    directories_to_keep: Collection[str] | None = None,
    group_indices_to_keep: Collection[int] | None = None,
) -> set[int] | None:
    """
    Resolve the final set of blendShape group (weight) indices to export,
    given directory-name and/or explicit-target filters.

    Returns ``None`` if no filtering was requested, meaning all groups
    are needed.
    """
    if directories_to_keep is None and group_indices_to_keep is None:
        return None
    needed_indices = _compute_needed_indices(
        directory_data,
        directories_to_keep=directories_to_keep or set(),
        group_indices_to_keep=group_indices_to_keep or set(),
    )
    resolved = {index for index in needed_indices if index >= 0}
    resolved.update(group_indices_to_keep or set())
    return resolved


def get_blendshape_directory_dict(
    blendshape: BlendShape,
) -> dict[int, BlendShapeTargetDirectory]:
    indices = blendshape.target_directory.get_indices()
    directory_dict: dict[int, BlendShapeTargetDirectory] = {}
    for index in indices:
        target_directory_attr = blendshape.target_directory[index]
        directory = BlendShapeTargetDirectory(
            name=target_directory_attr.dirctory_name.get(),
            parent_index=target_directory_attr.parent_index.get(),
            child_indices=target_directory_attr.child_indices.get(),
        )
        directory_dict[index] = directory
    return directory_dict


def get_blendshape_data(
    blendshape: str | BlendShape,
    directories: Collection[str] | None = None,
    targets: Iterable[str | int] | None = None,
) -> BlendShapeData:
    blendshape_node = (
        blendshape if isinstance(blendshape, BlendShape) else BlendShape.from_existing(blendshape)
    )
    target_groups_to_export = (
        {resolve_target_index(str(blendshape), target) for target in targets}
        if targets is not None
        else None
    )
    input_data = get_blendshape_input_data_list(blendshape_node)
    full_directory_data = get_blendshape_directory_dict(blendshape_node)
    needed_group_indices = resolve_needed_group_indices(
        full_directory_data, directories, target_groups_to_export
    )
    directory_data = (
        full_directory_data
        if needed_group_indices is None
        else _prune_blendshape_directory_dict(
            full_directory_data,
            directories_to_keep=directories or set(),
            group_indices_to_keep=needed_group_indices,
        )
    )
    target_data = get_blendshape_target_dict(blendshape_node, needed_group_indices)

    return BlendShapeData(
        blendshape_node.name,
        inputs={index: data for index, data in enumerate(input_data)},
        directory=directory_data,
        targets=target_data,
    )
