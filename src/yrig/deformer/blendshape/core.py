from collections.abc import Iterable
from pathlib import Path

from maya import cmds
from maya.api.OpenMaya import (
    MPlug,
    MSelectionList,
)

from yrig.select import maintain_selection


def create_blendshape(
    geometry: str,
    name: str,
    front_of_chain: bool = True,
) -> str:
    result: list[str] = cmds.blendShape(geometry, name=name, frontOfChain=front_of_chain)  # type: ignore
    if not result:
        raise RuntimeError(f"Failed to create blendShape: {name}")
    return result[0]


def import_blendshape(
    filepath: Path,
    blendshape: str | None = None,
) -> list[str]:
    """
    Create and load a blendShape from a .shape/.shp file.
    Args:
        filepath: Path to the exported shape file.
        blendshape_name: Optionally specifiy the name of the blendShape node to import to.
    Returns:
        The blendShape node(s) imported to.
    """

    if not filepath.exists():
        raise FileNotFoundError(f"Shape file does not exist: {filepath}")

    # Import into an existing blendShape.
    if blendshape and cmds.objExists(blendshape):
        cmds.blendShape(
            blendshape,
            edit=True,
            ip=str(filepath),
        )
        return [blendshape]

    # Let Maya create the blendShape from the file.
    kwargs = {}
    if blendshape:
        kwargs["name"] = blendshape

    with maintain_selection():
        cmds.select(clear=True)
        return cmds.blendShape(  # type: ignore
            **kwargs,  # type: ignore
            ip=str(filepath),
        )


def export_blendshape(
    filepath: Path,
    blendshape: str,
    targets: Iterable[str | int] | None = None,
) -> None:
    """
    Export a blendShape node to a .shape/.shp file.
    Args:
        blendshape_node: Name of the blendShape node.
        filepath: Output file path.
        targets: The names or indices of the targets to export,
    """

    if not cmds.objExists(blendshape):
        raise RuntimeError(f"BlendShape does not exist: {blendshape}")

    if cmds.nodeType(blendshape) != "blendShape":
        raise TypeError(f"{blendshape} is not a blendShape node")

    # Ensure output directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if targets is not None:
        target_indices = [resolve_target_index(blendshape, target) for target in targets]
        kwargs["exportTarget"] = [(0, index) for index in target_indices]
    # Export shape file
    cmds.blendShape(
        blendshape,
        **kwargs,  # type: ignore
        edit=True,
        export=str(filepath),
    )


def get_blendshape_targets(blendshape: str) -> dict[str, str]:
    """Return a mapping of blendShape target names to weight attributes.

    Args:
        blendshape: Name of the blendShape node.

    Returns:
        A mapping of target names to their corresponding weight attributes.
    """
    aliases = cmds.aliasAttr(blendshape, query=True) or []
    return dict(zip(aliases[::2], aliases[1::2], strict=True))


def get_target_index(blendshape: str, target: str) -> int:
    aliases = cmds.aliasAttr(blendshape, query=True) or []
    indices = cmds.getAttr(f"{blendshape}.weight", multiIndices=True) or []

    for alias, index in zip(aliases[::2], indices, strict=True):
        if alias == target:
            return index

    raise ValueError(f"Target {target} not found on {blendshape}.")


def resolve_target_index(blendshape: str, target: str | int) -> int:
    return target if isinstance(target, int) else get_target_index(blendshape, target)


def build_blendshape_networks(blendshape: str) -> dict[str, str]:
    """
    Creates one network node per blendshape type and connects
    custom attributes to the corresponding blendshape targets.

    Example:
        mouth_l_up_07
            -> mouth_l_nw.up_07

        mouth_l_up_07_out_10
            -> mouth_l_nw.up_07_out_10
    """
    network_nodes: dict[str, str] = {}

    for alias, attr in get_blendshape_targets(blendshape).items():
        parts = alias.split("_")

        if len(parts) < 3:
            cmds.warning(f"Invalid blendshape name: {alias}")
            continue

        target_type = "_".join(parts[:2])
        target_name = "_".join(parts[2:])

        network_name = f"{target_type}_nw"

        if target_type not in network_nodes:
            if cmds.objExists(network_name):
                network_node = network_name
            else:
                network_node = cmds.createNode(
                    "network",
                    name=network_name,
                )

            network_nodes[target_type] = network_node

        network_node = network_nodes[target_type]

        if not cmds.attributeQuery(
            target_name,
            node=network_node,
            exists=True,
        ):
            cmds.addAttr(
                network_node,
                longName=target_name,
                attributeType="double",
                defaultValue=0.0,
                minValue=0.0,
                maxValue=1.0,
                keyable=True,
            )

        cmds.connectAttr(
            f"{network_node}.{target_name}",
            f"{blendshape}.{attr}",
            force=True,
        )

    return network_nodes


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
