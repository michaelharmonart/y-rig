from collections.abc import Collection, Iterable

from maya.api.OpenMaya import (
    MDagPath,
    MFnComponent,
    MFnComponentListData,
    MFnPointArrayData,
    MObject,
    MPlug,
    MPoint,
    MPointArray,
)

from yrig.deformer.blendshape.core import (
    get_existing_target_group_indices,
    get_target_index_to_name_map,
    resolve_target_index,
)
from yrig.maya_api.attribute import (
    BlendShapeInputTargetAttribute,
    BlendShapeInputTargetGroupAttribute,
    BlendShapeInputTargetItemAttribute,
)
from yrig.maya_api.node import BlendShapeNode
from yrig.maya_api.utils import get_dag_path, get_plug
from yrig.shape import get_component_mfn_and_type_of_shape

from .data import (
    BlendShapeData,
    BlendShapeInputData,
    BlendShapeInputTargetData,
    BlendShapeTargetDirectory,
    BlendShapeTargetGroupData,
    BlendShapeTargetItemData,
)
from .directory import (
    prune_blendshape_directory_dict,
    resolve_needed_group_indices,
)

ZERO_TOLERANCE = 1e-7  # About the precision needed for 32 bit floats


def _filtered_components_and_points(
    components: list[int | tuple[int, int]], points_array: MPointArray
) -> tuple[list[int | tuple[int, int]], list[tuple[float, float, float]]]:
    """Returns a tuple of a list of filtered component indices, and a list of point tuples."""
    filtered_components: list[int | tuple[int, int]] = []
    filtered_point_tuples: list[tuple[float, float, float]] = []
    for component, point in zip(components, points_array, strict=True):  # type: ignore
        if not point.isEquivalent(MPoint.kOrigin, ZERO_TOLERANCE):
            filtered_components.append(component)
            filtered_point_tuples.append((point.x, point.y, point.z))
    return filtered_components, filtered_point_tuples


def get_component_indices(
    plug: MPlug, mfn_component_type: type[MFnComponent]
) -> list[int | tuple[int, int]]:
    components_mob: MObject = plug.asMObject()
    fn_components: MFnComponentListData = MFnComponentListData(components_mob)
    component_ids: list[int | tuple[int, int]] = []
    for x in range(fn_components.length()):
        comp_mob = fn_components.get(x)
        fn_comp = mfn_component_type(comp_mob)
        component_ids.extend(
            tuple(element) if not isinstance(element, int) else element
            for element in fn_comp.getElements()
        )
    return component_ids


def get_blendshape_target_item_data(
    input_geometry: MDagPath,
    target_item: BlendShapeInputTargetItemAttribute,
) -> BlendShapeTargetItemData:
    components_plug = get_plug(target_item.input_components_target)
    component_ids = get_component_indices(
        components_plug, get_component_mfn_and_type_of_shape(input_geometry)[0]
    )

    points_plug = get_plug(target_item.input_points_target)
    points_mob: MObject = points_plug.asMObject()
    fn_points: MFnPointArrayData = MFnPointArrayData(points_mob)
    points_array: MPointArray = fn_points.array()

    filtered_components, filtered_point_tuples = _filtered_components_and_points(
        component_ids, points_array
    )

    return BlendShapeTargetItemData(
        components=filtered_components,
        points=filtered_point_tuples,
    )


def get_blendshape_target_items_dict(
    input_geometry: MDagPath,
    target_group: BlendShapeInputTargetGroupAttribute,
) -> dict[int, BlendShapeTargetItemData]:
    items: dict[int, BlendShapeTargetItemData] = {}
    for index in target_group.input_target_item.get_indices():
        item = target_group.input_target_item[index]
        item_data = get_blendshape_target_item_data(input_geometry, item)
        items[index] = item_data
    return items


def get_blendshape_target_groups_dict(
    blendshape: BlendShapeNode,
    input_geometry: MDagPath,
    target: BlendShapeInputTargetAttribute,
    target_group_indices: Iterable[int] | None = None,
) -> dict[int, BlendShapeTargetGroupData]:
    target_groups: dict[int, BlendShapeTargetGroupData] = {}
    alias_map = get_target_index_to_name_map(str(blendshape))
    indices = (
        target_group_indices
        if target_group_indices is not None
        else target.input_target_group.get_indices()
    )
    for index in indices:
        target_group = target.input_target_group[index]
        target_name = alias_map[index]
        target_mode = target_group.post_deformers_mode.get()
        target_group_data = BlendShapeTargetGroupData(
            name=target_name,
            mode=target_mode,
            items=get_blendshape_target_items_dict(input_geometry, target_group),
        )
        target_groups[index] = target_group_data

    return target_groups


def get_blendshape_target_dict(
    blendshape: BlendShapeNode, target_group_indices: Iterable[int] | None = None
) -> dict[int, BlendShapeInputTargetData]:
    target_data_dict: dict[int, BlendShapeInputTargetData] = {}
    indices = blendshape.input.get_indices()
    for index in indices:
        input_geometry_attr = blendshape.input[index].input_geometry.get_input()
        if input_geometry_attr is None:
            raise RuntimeError(f"{input_geometry_attr} had no input geometry.")
        input_geometry_node = str(input_geometry_attr).split(".", 1)[0]
        input_geometry = get_dag_path(input_geometry_node)
        target = blendshape.input_target[index]
        group_data = get_blendshape_target_groups_dict(
            blendshape, input_geometry, target, target_group_indices
        )
        target_data = BlendShapeInputTargetData(groups=group_data)
        if target_data.groups:
            target_data_dict[index] = target_data
    return target_data_dict


def get_blendshape_input_data_list(blendshape: BlendShapeNode) -> list[BlendShapeInputData]:
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


def get_blendshape_directory_dict(
    blendshape: BlendShapeNode,
) -> dict[int, BlendShapeTargetDirectory]:
    indices = blendshape.target_directory.get_indices()
    existing_group_indices = get_existing_target_group_indices(blendshape)
    valid_indices = {-index for index in indices if index != 0} | existing_group_indices
    directory_dict: dict[int, BlendShapeTargetDirectory] = {}
    for index in indices:
        target_directory_attr = blendshape.target_directory[index]
        # Maya is stupid and sometimes creates child index lists that have duplicates...
        child_indices = list(dict.fromkeys(target_directory_attr.child_indices.get()))
        valid_child_indices = [
            child_index for child_index in child_indices if child_index in valid_indices
        ]
        directory = BlendShapeTargetDirectory(
            name=target_directory_attr.directory_name.get(),
            parent_index=target_directory_attr.parent_index.get(),
            child_indices=valid_child_indices,
        )
        directory_dict[index] = directory
    return directory_dict


def get_blendshape_data(
    blendshape: str | BlendShapeNode,
    directories: Collection[str] | None = None,
    targets: Iterable[str | int] | None = None,
) -> BlendShapeData:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    target_groups_to_export = (
        {resolve_target_index(str(blendshape), target) for target in targets}
        if targets is not None
        else None
    )
    input_data = get_blendshape_input_data_list(blendshape_node)

    full_directory_data = get_blendshape_directory_dict(blendshape_node)

    needed_group_indices = resolve_needed_group_indices(
        full_directory_data, directories, target_groups_to_export, existing_group_indices
    )
    directory_data = (
        full_directory_data
        if needed_group_indices is None
        else prune_blendshape_directory_dict(
            full_directory_data,
            existing_group_indices=existing_group_indices,
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
