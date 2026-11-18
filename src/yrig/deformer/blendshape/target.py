from maya import cmds
from maya.api.OpenMaya import MPlug, MSelectionList

from yrig.deformer.blendshape.core import get_blendshape_input_geometry_indices
from yrig.maya_api.node import BlendShapeNode
from yrig.shape import get_shape, get_shape_output


def get_blendshape_targets(blendshape: str) -> dict[str, str]:
    """Return a mapping of blendShape target names to weight attributes.

    Args:
        blendshape: Name of the blendShape node.

    Returns:
        A mapping of target names to their corresponding weight attributes.
    """
    aliases = cmds.aliasAttr(blendshape, query=True) or []
    return dict(zip(aliases[::2], aliases[1::2], strict=True))


def get_existing_target_group_indices(blendshape: BlendShapeNode) -> set[int]:
    existing_group_indices: set[int] = set()
    for index in blendshape.input_target.get_indices():
        input_target = blendshape.input_target[index]
        target_group_indices = input_target.input_target_group.get_indices()
        existing_group_indices.update(target_group_indices)
    return existing_group_indices


def get_target_index(blendshape: str, target: str) -> int:
    aliases = cmds.aliasAttr(blendshape, query=True) or []
    indices = cmds.getAttr(f"{blendshape}.weight", multiIndices=True) or []

    for alias, index in zip(aliases[::2], indices, strict=True):
        if alias == target:
            return index

    raise ValueError(f"Target {target} not found on {blendshape}.")


def resolve_target_index(blendshape: str, target: str | int) -> int:
    return target if isinstance(target, int) else get_target_index(blendshape, target)


def get_target_index_to_name_map(blendshape: str) -> dict[int, str]:
    aliases = cmds.aliasAttr(blendshape, query=True) or []
    return {
        int(attr.removeprefix("weight[").removesuffix("]")): alias
        for alias, attr in zip(aliases[::2], aliases[1::2], strict=True)
    }


def get_name_to_target_index_map(blendshape: str) -> dict[str, int]:
    aliases = cmds.aliasAttr(blendshape, query=True) or []
    return {
        alias: int(attr.removeprefix("weight[").removesuffix("]"))
        for alias, attr in zip(aliases[::2], aliases[1::2], strict=True)
    }


def get_blendshape_target_names(blendshape: str) -> list[str]:
    """Return all target names on a blendShape node."""
    aliases = cmds.aliasAttr(blendshape, query=True) or []
    return aliases[::2]


def get_target_weights(
    blendshape: str, target: str | int, input_index: int = 0
) -> dict[int, float]:
    """
    Return the stored vertex weights for a blendShape target.

    Args:
        blendshape: Name of the blendShape node.
        target: Target alias name or logical target index.
        input_index: Index of the input geometry.

    Returns: A mapping of vertex indices to their target weights. Only vertices
        with explicitly stored weights are included.
    """
    resolved_target_index = resolve_target_index(blendshape, target)
    target_attr = (
        f"{blendshape}.inputTarget[{input_index}].inputTargetGroup[{resolved_target_index}]"
    )

    sel: MSelectionList = MSelectionList()
    sel.add(f"{target_attr}.targetWeights")
    weight_list_plug: MPlug = sel.getPlug(0)
    indices = weight_list_plug.getExistingArrayAttributeIndices()
    weights_dict: dict[int, float] = {}
    for i in indices:
        weight_plug: MPlug = weight_list_plug.elementByLogicalIndex(i)
        value = weight_plug.asDouble()
        weights_dict[i] = value

    return weights_dict


def set_target_weights(
    blendshape: str,
    target: str | int,
    weights: dict[int, float],
    input_index: int = 0,
    clear_existing: bool = True,
) -> None:
    """Set the stored vertex weights for a blendShape target.

    Args:
        blendshape: Name of the blendShape node.
        target: Target alias name or logical target index.
        weights: Mapping of vertex indices to their target weights.
        input_index: Index of the input geometry.
    """
    resolved_target_index = resolve_target_index(blendshape, target)
    target_attr = (
        f"{blendshape}.inputTarget[{input_index}].inputTargetGroup[{resolved_target_index}]"
    )

    sel: MSelectionList = MSelectionList()
    sel.add(f"{target_attr}.targetWeights")
    weight_list_plug: MPlug = sel.getPlug(0)

    if clear_existing:
        num_points = weight_list_plug.numElements()
        for i in range(num_points):
            weight_plug: MPlug = weight_list_plug.elementByPhysicalIndex(i)
            logical_index = weight_plug.logicalIndex()
            if logical_index not in weights:
                weight_plug.setDouble(0)

    for index, value in weights.items():
        weight_plug: MPlug = weight_list_plug.elementByLogicalIndex(index)
        weight_plug.setDouble(value)


def connect_target_deltas(source_target_attr: str, driven_target_attr: str) -> None:
    source_target_item_attr = f"{source_target_attr}.inputTargetItem"
    driven_target_item_attr = f"{driven_target_attr}.inputTargetItem"
    source_target_item_indices = cmds.getAttr(source_target_item_attr, multiIndices=True) or []
    for index in source_target_item_indices:
        for attr in (
            "inputGeomTarget",
            "inputRelativePointsTarget",
            "inputRelativeComponentsTarget",
            "inputPointsTarget",
            "inputComponentsTarget",
        ):
            cmds.connectAttr(
                f"{source_target_item_attr}[{index}].{attr}",
                f"{driven_target_item_attr}[{index}].{attr}",
            )


def copy_target_deltas(
    blendshape: BlendShapeNode | str,
    source_index: int,
    destination_index: int,
    *,
    input_index: int | None = None,
    replace: bool = True,
) -> None:
    for index in (
        (input_index,)
        if input_index is not None
        else get_blendshape_input_geometry_indices(blendshape)
    ):
        if replace:
            cmds.blendShape(str(blendshape), edit=True, resetTargetDelta=(index, destination_index))
        cmds.blendShape(
            str(blendshape),
            edit=True,
            copyDelta=(index, source_index, destination_index),
        )


def reset_target_delta(
    blendshape: BlendShapeNode | str,
    target_index: int,
    *,
    input_index: int | None = None,
) -> None:
    for index in (
        (input_index,)
        if input_index is not None
        else get_blendshape_input_geometry_indices(blendshape)
    ):
        cmds.blendShape(str(blendshape), edit=True, resetTargetDelta=(index, target_index))


def add_blendshape_target_from_geometry(
    blendshape: str | BlendShapeNode,
    geometry: str,
    name: str | None = None,
    default_value: float = 1,
    *,
    local: bool = True,
) -> int:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    index = blendshape_node.input_target[0].input_target_group.next_available_index()
    target_item = blendshape_node.input_target[0].input_target_group[index].input_target_item[6000]

    shape = get_shape(geometry)
    if shape is None:
        raise RuntimeError(f"Couldn't find shape for {geometry}.")
    shape_out = get_shape_output(shape, local)
    target_item.input_geom_target.connect_from(shape_out)
    blendshape_node.weight[index].set_alias(name or geometry)
    blendshape_node.weight[index].set(default_value)
    return index
