from pathlib import PurePosixPath

from maya import cmds
from maya.api.OpenMaya import MDagPath, MFn
from maya.api.OpenMayaAnim import MFnGeometryFilter

from yrig.maya_api.node import BlendShapeNode, ShapeEditorManagerNode
from yrig.maya_api.utils import get_depend_node

from .directory import (
    add_blendshape_to_directory,
    add_shape_editor_directory,
    find_child_directory_index,
    preserve_shape_editor_root_directory,
)


def resolve_blendshape(node: str) -> BlendShapeNode | None:
    if cmds.objectType(node, isAType="blendShape"):
        return BlendShapeNode.from_existing(node)
    history: list[str] = cmds.listHistory(node)  # type: ignore
    blendshapes = cmds.ls(history, type="blendShape")  # type: ignore
    return BlendShapeNode.from_existing(blendshapes[0]) if blendshapes else None


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
    with preserve_shape_editor_root_directory(shape_editor_manager):
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


def get_blendshape_input_geometry_indices(blendshape: str | BlendShapeNode) -> list[int]:
    result: list[int] = cmds.blendShape(str(blendshape), query=True, geometryIndices=True) or []  # type: ignore
    return result


def get_blendshape_input_shape(
    blendshape: BlendShapeNode,
    input_index: int,
) -> str | None:
    """Return the upstream DAG shape for an existing logical input index."""
    if isinstance(input_index, bool) or not isinstance(input_index, int):
        raise TypeError("input_index must be an integer")
    if input_index < 0:
        raise ValueError("input_index must be nonnegative")

    node = get_depend_node(str(blendshape))
    try:
        shape = MFnGeometryFilter(node).inputShapeAtIndex(input_index)
    except RuntimeError as exc:
        return None
    if shape.isNull() or not shape.hasFn(MFn.kShape):
        return None
    shape_path: MDagPath = MDagPath.getAPathTo(shape)
    return shape_path.partialPathName()


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


def add_blendshape_target(
    blendshape: str | BlendShapeNode,
    name: str,
    parent_directory_index: int = 0,
) -> int:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    parent_directory = blendshape_node.target_directory[parent_directory_index]
    original_child_indices = parent_directory.child_indices.get()
    target_group_index = blendshape_node.weight.next_available_index()
    target_attr = blendshape_node.weight[target_group_index]
    target_attr.set_alias(name)
    target_attr.set(1)
    parent_directory.child_indices.set([*original_child_indices, target_group_index])
    return target_group_index
