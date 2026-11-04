from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from maya import cmds
from maya.api.OpenMaya import (
    MPlug,
    MSelectionList,
)

from yrig.maya_api.node import BlendShapeNode, ShapeEditorManagerNode
from yrig.select import maintain_selection
from yrig.shape import get_shape, get_shape_output


@contextmanager
def preserve_all_shape_editor_directories(
    manager: ShapeEditorManagerNode,
) -> Generator[None, None, None]:
    """
    When initializing a blendShape Maya just decides a directory to add it to.
    This screws up our manual handling of directories on import.

    This might be overkill, but since I don't have the Maya source code on hand
    this is an easy way to guarantee that Maya won't screw up directory data.
    """
    directory_indices = manager.blend_shape_directory.get_indices()
    snapshot = {
        index: manager.blend_shape_directory[index].child_indices.get()
        for index in directory_indices
    }
    try:
        yield
    finally:
        for index, child_indices in snapshot.items():
            manager.blend_shape_directory[index].child_indices.set(child_indices)


def find_child_directory_index(
    manager: ShapeEditorManagerNode,
    parent_directory_index: int,
    name: str,
) -> int | None:
    """Return the index of the direct child directory called ``name``, or None."""
    child_indices = manager.blend_shape_directory[parent_directory_index].child_indices.get()
    for child_index in child_indices:
        if child_index >= 0:  # positive entries are non directory children
            continue
        directory_index = -child_index
        directory = manager.blend_shape_directory[directory_index]
        if directory.directory_name.get() == name:
            return directory_index
    return None


def add_blendshape_to_directory(
    manager: ShapeEditorManagerNode, blendshape: BlendShapeNode, parent_directory_index: int
) -> int:
    child_index = blendshape.mid_layer_id.get()
    with preserve_all_shape_editor_directories(manager):
        blendshape.mid_layer_parent.set(parent_directory_index)
    parent_directory = manager.blend_shape_directory[parent_directory_index]
    original_child_indices = parent_directory.child_indices.get()
    parent_directory.child_indices.set(value=[*original_child_indices, child_index])

    return child_index


def add_shape_editor_directory(
    manager: ShapeEditorManagerNode, name: str, parent_directory_index: int
) -> int:
    directory_index = manager.blend_shape_directory.next_available_index()
    directory = manager.blend_shape_directory[directory_index]
    with preserve_all_shape_editor_directories(manager):
        directory.directory_name.set(name)
        directory.parent_index.set(parent_directory_index)
    parent_directory = manager.blend_shape_directory[parent_directory_index]
    original_child_indices = parent_directory.child_indices.get()
    parent_directory.child_indices.set([*original_child_indices, -directory_index])
    return directory_index


def create_blendshape(
    geometry: str,
    name: str | None = None,
    *,
    front_of_chain: bool = True,
    parent_directory: PurePosixPath | str = "/",
) -> BlendShapeNode:
    resolved_name = name or f"{geometry}_BS"

    parent_directory_path = PurePosixPath(parent_directory)
    parent_directories = tuple(
        part for part in parent_directory_path.parts if part != parent_directory_path.anchor
    )
    shape_editor_manager = ShapeEditorManagerNode.from_existing("shapeEditorManager")
    with preserve_all_shape_editor_directories(shape_editor_manager):
        maya_result: list[str] = cmds.blendShape(
            geometry, name=resolved_name, frontOfChain=front_of_chain
        )  # type: ignore
    if not maya_result:
        raise RuntimeError(f"Failed to create blendShape: {resolved_name}")
    blendshape_node = BlendShapeNode.from_existing(maya_result[0])
    parent_index: int = 0
    for dir in parent_directories:
        child_index: int | None = find_child_directory_index(
            shape_editor_manager, parent_index, dir
        )
        if child_index is None:
            parent_index = add_shape_editor_directory(shape_editor_manager, dir, parent_index)
        else:
            parent_index = child_index
    add_blendshape_to_directory(shape_editor_manager, blendshape_node, parent_index)
    return blendshape_node


def import_maya_shape_file(
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


def export_maya_shape_file(
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
    if targets:
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
    input_index: int,
    source_index: int,
    destination_index: int,
    *,
    replace: bool = True,
) -> None:
    if replace:
        cmds.blendShape(
            str(blendshape), edit=True, resetTargetDelta=(input_index, destination_index)
        )
    cmds.blendShape(
        str(blendshape),
        edit=True,
        copyDelta=(input_index, source_index, destination_index),
    )


def connect_to_psd_network(
    blendshape: str | BlendShapeNode, psd_network: str, create_missing: bool = True
) -> tuple[str, list[str]]:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    if cmds.objExists(psd_network):
        resolved_psd_network = psd_network
    else:
        if create_missing:
            resolved_psd_network = cmds.createNode("network", name=psd_network)
            cmds.setAttr(f"{resolved_psd_network}.isHistoricallyInteresting", False)  # type: ignore
        else:
            raise RuntimeError(f"Couldn't find the node {psd_network}")
    connected: list[str] = []
    for weight_attr in blendshape_node.weight:
        alias = weight_attr.get_alias()
        if alias is not None:
            input = weight_attr.get_input()
            if not input:
                network_alias_attr = f"{resolved_psd_network}.{alias}"
                if not cmds.objExists(network_alias_attr):
                    if create_missing:
                        cmds.addAttr(resolved_psd_network, longName=alias, attributeType="float")
                    else:
                        continue
                weight_attr.connect_from(network_alias_attr)
                connected.append(network_alias_attr)
    return resolved_psd_network, connected


def get_existing_target_group_indices(blendshape: BlendShapeNode) -> set[int]:
    existing_group_indices: set[int] = set()
    for index in blendshape.input_target.get_indices():
        input_target = blendshape.input_target[index]
        target_group_indices = input_target.input_target_group.get_indices()
        existing_group_indices.update(target_group_indices)
    return existing_group_indices


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
