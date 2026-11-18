from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from maya import cmds
from maya.api.OpenMaya import (
    MDagPath,
    MFnPointArrayData,
    MObject,
    MPlug,
    MPointArray,
    MSelectionList,
)

if TYPE_CHECKING:
    from .attribute import Attribute
    from .node import Node

log = logging.getLogger(__name__)

_loaded_plugin_cache: set[str] = set()


def ensure_plugin_loaded(plugin: str) -> None:
    if plugin not in _loaded_plugin_cache:
        if not cmds.pluginInfo(plugin, query=True, loaded=True):
            cmds.loadPlugin(plugin)
            log.info(f"Loaded plugin: {plugin}")
        _loaded_plugin_cache.add(plugin)


def list_attribute_origins(node: str) -> None:
    """
    Print each attribute and the Maya node type where it is first defined.
    Use this when adding new node wrappers.
    """
    node_type = cmds.nodeType(node)

    # Root-first chain, then the node's own type last.
    type_chain: list[str] = [*cmds.nodeType(node, inherited=True), node_type]  # type: ignore

    print(f"\n{node} ({node_type})")  # noqa
    print("=" * 80)  # noqa

    seen: set[str] = set()
    pending: list[str] = []  # types attributeInfo can't resolve

    for level, type_name in enumerate(type_chain):
        try:
            attrs = cmds.attributeInfo(type=type_name, allAttributes=True) or []
        except RuntimeError:
            pending.append(type_name)
            continue

        new_attrs = sorted(set(attrs) - seen)
        seen.update(attrs)

        label = " + ".join([*pending, type_name])
        pending.clear()

        if not new_attrs:
            continue

        print(f"\nLevel {level}: {label}")  # noqa
        print("-" * 80)  # noqa
        for attr in new_attrs:
            print(f"{attr:<40} {type_name}")  # noqa


def get_dag_path(node: str | Node) -> MDagPath:
    node_str = str(node)
    selection = MSelectionList()
    try:
        selection.add(node)
        dag_path: MDagPath = selection.getDagPath(0)
    except RuntimeError as exc:
        found_nodes = cmds.ls(node_str)
        if found_nodes:
            raise RuntimeError(
                f"Couldn't resolve an MDagPath for '{node}' as there were multiple nodes with that name: "
                f"{', '.join(found_nodes)}"
            ) from exc
        else:
            raise RuntimeError(
                f"Couldn't resolve an MDagPath for {node} as it wasn't present in the scene."
            ) from exc
    except TypeError as exc:
        raise TypeError(
            f"Couldn't resolve an MDagPath for {node} as it is an object of the wrong type."
        ) from exc
    return dag_path


def get_depend_node(node: str | Node) -> MObject:
    node_str = str(node)
    selection = MSelectionList()
    try:
        selection.add(node_str)
        depend_node: MObject = selection.getDependNode(0)
    except RuntimeError as exc:
        raise RuntimeError(f"Couldn't resolve an MObject for {node}") from exc
    return depend_node


def get_plug(attr: str | Attribute) -> MPlug:
    attr_str = str(attr)
    selection = MSelectionList()
    try:
        selection.add(attr_str)
        plug: MPlug = selection.getPlug(0)
    except RuntimeError as exc:
        raise RuntimeError(f"Couldn't resolve an MPlug for {attr}") from exc
    return plug


def set_point_array(plug: MPlug, point_array: MPointArray) -> None:
    fn_points = MFnPointArrayData()
    points_mob = fn_points.create(point_array)
    plug.setMObject(points_mob)
