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

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from maya import cmds
from maya.api.OpenMaya import (
    MFn,
    MFnComponentListData,
    MFnPointArrayData,
    MFnSingleIndexedComponent,
    MObject,
    MPlug,
    MPoint,
    MPointArray,
)

from yrig.deformer.blendshape.core import resolve_target_index
from yrig.io import confirm_overwrite
from yrig.io.json import export_json, load_json
from yrig.maya_api.attribute import (
    BlendShapeInputTargetAttribute,
    BlendShapeInputTargetGroupAttribute,
    BlendShapeInputTargetItemAttribute,
)
from yrig.maya_api.node import BlendShape
from yrig.maya_api.utils import get_plug


@dataclass
class BlendShapeData:
    name: str
    inputs: dict[int, BlendShapeInputData]
    directory: dict[int, BlendShapeTargetDirectory]
    targets: dict[int, BlendShapeInputTargetData]


@dataclass
class BlendShapeInputData:
    geometry: str
    original_geometry: str | None
    group_id: int = 0
    component_tag_expression: str = "*"


@dataclass
class BlendShapeTargetDirectory:
    name: str
    parent_index: int
    child_indices: list[int]


@dataclass
class BlendShapeInputTargetData:
    groups: dict[int, BlendShapeTargetGroupData]


@dataclass
class BlendShapeTargetGroupData:
    name: str
    items: dict[int, BlendShapeTargetItemData]


@dataclass
class BlendShapeTargetItemData:
    components: list[int]
    points: list[tuple[float, float, float]]


def _get_component_indices(plug: MPlug) -> list[int]:
    components_mob: MObject = plug.asMObject()
    fn_components: MFnComponentListData = MFnComponentListData(components_mob)
    component_ids: list[int] = []
    for x in range(fn_components.length()):
        comp_mob = fn_components.get(x)
        fn_comp = MFnSingleIndexedComponent(comp_mob)
        component_ids.extend(fn_comp.getElements())
    return component_ids


def _set_component_indices(plug: MPlug, indices: list[int]) -> None:
    fn_data: MFnComponentListData = MFnComponentListData()
    data_mob: MObject = fn_data.create()
    fn_comp: MFnSingleIndexedComponent = MFnSingleIndexedComponent()
    comp_mob: MObject = fn_comp.create(MFn.kMeshVertComponent)
    fn_comp.addElements(indices)
    fn_data.add(comp_mob)
    plug.setMObject(data_mob)


def _set_point_array(plug: MPlug, point_array: MPointArray) -> None:
    fn_points = MFnPointArrayData()
    points_mob = fn_points.create(point_array)
    plug.setMObject(points_mob)


def get_blendshape_target_item_data(
    target_item: BlendShapeInputTargetItemAttribute,
) -> BlendShapeTargetItemData:
    components_plug = get_plug(str(target_item.input_components_target))
    component_ids = _get_component_indices(components_plug)

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

    _set_component_indices(component_plug, data.components)
    _set_point_array(points_plug, point_array)


def get_blendshape_target_items_dict(
    target_group: BlendShapeInputTargetGroupAttribute,
) -> dict[int, BlendShapeTargetItemData]:
    items: dict[int, BlendShapeTargetItemData] = {}
    for index in target_group.input_target_item.get_indices():
        item = target_group.input_target_item[index]
        item_data = get_blendshape_target_item_data(item)
        items[index] = item_data
    return items


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


def get_target_name_map(blendshape: BlendShape) -> dict[int, str]:
    aliases = cmds.aliasAttr(str(blendshape), query=True) or []
    return {
        int(attr.removeprefix("weight[").removesuffix("]")): alias
        for alias, attr in zip(aliases[::2], aliases[1::2], strict=True)
    }


def get_blendshape_target_groups_dict(
    blendshape: BlendShape,
    target: BlendShapeInputTargetAttribute,
    target_group_indices: Iterable[int] | None = None,
) -> dict[int, BlendShapeTargetGroupData]:
    target_groups: dict[int, BlendShapeTargetGroupData] = {}
    alias_map = get_target_name_map(blendshape)
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


def apply_blendshape_data(
    blendshape: str | BlendShape, data: BlendShapeData, targets: Collection[str] | None = None
) -> None:
    blendshape_node = (
        blendshape if isinstance(blendshape, BlendShape) else BlendShape.from_existing(blendshape)
    )
    for target_index, target in data.targets.items():
        for group_index, group in target.groups.items():
            if not targets or group.name in targets:
                apply_blendshape_target_group_data(
                    blendshape_node, group, target_index=target_index, group_index=group_index
                )


def export_blendshape(
    filepath: Path,
    blendshape: str | BlendShape,
    directories: Collection[str] | None = None,
    targets: Iterable[str | int] | None = None,
    force: bool = False,
) -> bool:
    """
    Export blendShape target data to a `.yshape` file.

    Args:
        filepath: Destination `.yshape` file.
        blendshape: BlendShape node to export.
        targets: Optional target names or indices to export.
        force: Whether to overwrite an existing file without confirmation.

    Returns:
        ``True`` if the blendShape was exported, or ``False``
        if the export was cancelled."""
    if filepath.suffix != ".yshape":
        raise ValueError("Blendshaspe files should use the .yshape extension.")
    if not confirm_overwrite(filepath, force):
        return False
    blendshape_data = get_blendshape_data(blendshape, directories, targets)
    export_json(filepath, blendshape_data)
    return True


def import_blendshape(
    filepath: Path,
    blendshape: str | BlendShape,
    targets: Collection[str] | None = None,
) -> None:
    """
    Import blendShape target data from a `.yshape` file.

    Args:
        filepath: Source `.yshape` file containing serialized blendShape data.
        blendshape: BlendShape node to modify.
        targets: Optional collection of target names to import. If omitted, all targets contained in the file are imported.
    """
    data = load_json(filepath, BlendShapeData)
    apply_blendshape_data(blendshape, data, targets)
