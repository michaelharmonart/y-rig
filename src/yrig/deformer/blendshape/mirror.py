import logging
from collections.abc import Callable, Iterable, Mapping
from typing import cast

from maya.api.OpenMaya import MDagPath
from mgear.core.utils import get_dag_path

from yrig.deformer.blendshape.core import (
    get_blendshape_input_shape,
    get_blendshape_target_names,
    get_name_to_target_index_map,
)
from yrig.deformer.blendshape.serialize.apply import apply_blendshape_target_group_data
from yrig.deformer.blendshape.serialize.data import (
    BlendShapeTargetGroupData,
    BlendShapeTargetItemData,
)
from yrig.deformer.blendshape.serialize.get import get_blendshape_target_group_data
from yrig.maya_api.enum import BlendShapePostDeformationOrder
from yrig.maya_api.node import BlendShapeNode
from yrig.mirror import (
    MirrorAxis,
    TopologyMirrorMap,
    generate_geometry_mirror_mapping,
)
from yrig.name import Side, filter_side, flip_side

log = logging.getLogger(__name__)

ComponentIndex = int | tuple[int, int]


def _mirror_delta(
    point: tuple[float, float, float],
    axis: MirrorAxis,
) -> tuple[float, float, float]:
    x, y, z = point

    if axis == MirrorAxis.X:
        return -x, y, z
    if axis == MirrorAxis.Y:
        return x, -y, z
    if axis == MirrorAxis.Z:
        return x, y, -z

    return point


def mirror_blendshape_target_item_data(
    data: BlendShapeTargetItemData,
    mirror_map: TopologyMirrorMap,
    mirror_axis: MirrorAxis = MirrorAxis.X,
) -> BlendShapeTargetItemData:
    """Mirror sparse deltas using a source-to-destination component map."""
    component_map = cast(
        Mapping[ComponentIndex, ComponentIndex],
        mirror_map,
    )
    mirrored_components: list[ComponentIndex] = []
    mirrored_points: list[tuple[float, float, float]] = []
    destination_sources: dict[ComponentIndex, ComponentIndex] = {}

    for source_component, point in zip(data.components, data.points, strict=True):
        destination = component_map[source_component]

        if destination in destination_sources:
            previous_source = destination_sources[destination]
            raise ValueError(
                f"Mirror map collision: sources {previous_source} and {source_component} "
                f"both map to destination {destination}."
            )

        destination_sources[destination] = source_component
        mirrored_components.append(destination)
        mirrored_points.append(_mirror_delta(point, mirror_axis))

    return BlendShapeTargetItemData(
        components=mirrored_components,
        points=mirrored_points,
    )


def mirror_blendshape_target_group_data(
    data: BlendShapeTargetGroupData,
    mirror_map: TopologyMirrorMap,
    mirror_axis: MirrorAxis = MirrorAxis.X,
) -> BlendShapeTargetGroupData:
    if data.mode == BlendShapePostDeformationOrder.TANGENT_SPACE:
        log.warning(
            "Blendshape mirroring currently has no logic to handle mirroring tangent space deltas! "
            "The mirrored shapes will likely look broken."
        )
    new_items = {
        index: mirror_blendshape_target_item_data(item, mirror_map, mirror_axis)
        for index, item in data.items.items()
    }
    return BlendShapeTargetGroupData(
        name=data.name,
        mode=data.mode,
        items=new_items,
    )


def mirror_blendshape_target(
    blendshape: BlendShapeNode,
    input_geometry: MDagPath,
    *,
    input_index: int,
    source_index: int,
    destination_index: int,
    mirror_axis: MirrorAxis = MirrorAxis.X,
    mirror_map: TopologyMirrorMap,
) -> None:
    input_target = blendshape.input_target[input_index]
    source_target_group = input_target.input_target_group[source_index]
    target_group_data = get_blendshape_target_group_data(
        blendshape, input_geometry, source_target_group, target_index=source_index
    )
    mirrored_data = mirror_blendshape_target_group_data(target_group_data, mirror_map, mirror_axis)
    apply_blendshape_target_group_data(
        blendshape,
        mirrored_data,
        input_geometry,
        input_index,
        destination_index,
        apply_name=False,
    )


def mirror_blendshape_targets(
    blendshape: BlendShapeNode | str,
    targets: Iterable[str] | str,
    *,
    mirror_axis: MirrorAxis = MirrorAxis.X,
    map_mirror_target: Callable[[str], str] | None = None,
) -> dict[str, str]:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    original_targets = [targets] if isinstance(targets, str) else list(targets)
    mirrored_targets = [
        map_mirror_target(target) if map_mirror_target is not None else flip_side(target)
        for target in original_targets
    ]
    name_to_index_map = get_name_to_target_index_map(str(blendshape_node))
    mirrored: dict[str, str] = {}
    input_indices = list(blendshape_node.input_target.get_indices())
    input_shape_map = {
        input_index: shape
        for input_index in input_indices
        if (shape := get_blendshape_input_shape(blendshape_node, input_index)) is not None
    }
    input_component_mirror_map = {}
    for input_index in input_indices:
        original_geometry_attr = blendshape_node.original_geometry[input_index].get_input()
        if original_geometry_attr is None:
            raise RuntimeError(
                f"No original geometry connected to "
                f"{blendshape_node}.originalGeometry[{input_index}]"
            )
        original_geometry = str(original_geometry_attr).split(".", 1)[0]
        input_component_mirror_map[input_index] = generate_geometry_mirror_mapping(
            original_geometry, mirror_axis
        )
        for original_target, mirrored_target in zip(
            original_targets, mirrored_targets, strict=True
        ):
            if original_target == mirrored_target:
                # TODO: Make this mirror the target itself (for things like jaw_open, that should be self symmetrical)
                continue
            original_target_index = name_to_index_map[original_target]
            if mirrored_target not in name_to_index_map:
                raise NotImplementedError(
                    "This should create a mirror target if it doesn't already exist. Ideally it would even put it in the right directory."
                )
            mirrored_target_index = name_to_index_map[mirrored_target]
            input_target = blendshape_node.input_target[input_index]
            input_shape: str = input_shape_map[input_index]
            input_geometry_path = get_dag_path(input_shape)
            existing_target_group_indices = set(input_target.input_target_group.get_indices())
            if original_target_index not in existing_target_group_indices:
                continue
            mirror_blendshape_target(
                blendshape_node,
                input_geometry=input_geometry_path,
                input_index=input_index,
                source_index=original_target_index,
                destination_index=mirrored_target_index,
                mirror_axis=mirror_axis,
                mirror_map=input_component_mirror_map[input_index],
            )
            mirrored[original_target] = mirrored_target
    return mirrored


def mirror_all_blendshape_targets(
    blendshape: BlendShapeNode | str,
    *,
    left_to_right: bool = True,
    map_mirror_target: Callable[[str], str] | None = None,
) -> dict[str, str]:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    source_side = Side.LEFT if left_to_right else Side.RIGHT

    targets = filter_side(get_blendshape_target_names(str(blendshape_node)), source_side)

    return mirror_blendshape_targets(
        blendshape_node,
        targets,
        map_mirror_target=map_mirror_target,
    )
