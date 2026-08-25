from pathlib import Path

from maya import cmds


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
    geometry: str,
    name: str,
) -> str:
    """
    Create and load a blendShape from a .shape/.shp file.
    Args:
        filepath: Path to the exported shape file.
        geometry: geometry the blendShape will deform.
        blendshape_name: Name of the blendShape node.
    Returns:
        The blendShape as a blendshape data class.
    """

    if not filepath.exists():
        raise FileNotFoundError(f"Shape file does not exist: {filepath}")

    # Reuse existing blendShape if it already exists
    if cmds.objExists(name):
        blendshape_node = name
    else:
        blendshape_node = create_blendshape(geometry, name)

    # Import shape data
    cmds.blendShape(
        blendshape_node,
        edit=True,
        ip=str(filepath),
    )

    return blendshape_node


def export_blendshape(
    blendshape_node: str,
    filepath: Path,
) -> None:
    """
    Export a blendShape node to a .shape/.shp file.
    Args:
        blendshape_node: Name of the blendShape node.
        filepath: Output file path.
    """

    if not cmds.objExists(blendshape_node):
        raise RuntimeError(f"BlendShape does not exist: {blendshape_node}")

    if cmds.nodeType(blendshape_node) != "blendShape":
        raise TypeError(f"{blendshape_node} is not a blendShape node")

    # Ensure output directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Export shape file
    cmds.blendShape(
        blendshape_node,
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
