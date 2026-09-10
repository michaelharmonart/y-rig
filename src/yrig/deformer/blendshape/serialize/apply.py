from __future__ import annotations

from collections.abc import Collection

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

from yrig.maya_api.node import BlendShape
from yrig.maya_api.utils import get_plug

from .data import BlendShapeData, BlendShapeTargetGroupData, BlendShapeTargetItemData


def _set_component_list_indices(plug: MPlug, indices: list[int]) -> None:
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

    _set_component_list_indices(component_plug, data.components)
    _set_point_array(points_plug, point_array)


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
