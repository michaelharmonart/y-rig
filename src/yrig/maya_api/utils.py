import logging

from maya import cmds
from maya.api.OpenMaya import (
    MDagPath,
    MFn,
    MFnComponentListData,
    MFnPointArrayData,
    MFnSingleIndexedComponent,
    MObject,
    MPlug,
    MPointArray,
    MSelectionList,
)

log = logging.getLogger(__name__)

_loaded_plugin_cache: set[str] = set()


def ensure_plugin_loaded(plugin: str) -> None:
    if plugin not in _loaded_plugin_cache:
        if not cmds.pluginInfo(plugin, query=True, loaded=True):
            cmds.loadPlugin(plugin)
            log.info(f"Loaded plugin: {plugin}")
        _loaded_plugin_cache.add(plugin)


def get_dag_path(node: str) -> MDagPath:
    selection = MSelectionList()
    try:
        selection.add(node)
        dag_path: MDagPath = selection.getDagPath(0)
    except RuntimeError as exc:
        found_nodes = cmds.ls(node)
        if found_nodes:
            raise RuntimeError(
                f"Couldn't resolve an MDagPath for '{node}' as there were multiple nodes with that name: "
                f"{', '.join(found_nodes)}"
            ) from exc
        else:
            raise RuntimeError(f"Couldn't resolve an MDagPath for {node}") from exc
    return dag_path


def get_depend_node(node: str) -> MObject:
    selection = MSelectionList()
    try:
        selection.add(node)
        depend_node: MObject = selection.getDependNode(0)
    except RuntimeError as exc:
        raise RuntimeError(f"Couldn't resolve an MObject for {node}") from exc
    return depend_node


def get_plug(attr: str) -> MPlug:
    selection = MSelectionList()
    try:
        selection.add(attr)
        plug: MPlug = selection.getPlug(0)
    except RuntimeError as exc:
        raise RuntimeError(f"Couldn't resolve an MPlug for {attr}") from exc
    return plug


def get_component_indices(plug: MPlug) -> list[int]:
    components_mob: MObject = plug.asMObject()
    fn_components: MFnComponentListData = MFnComponentListData(components_mob)
    component_ids: list[int] = []
    for x in range(fn_components.length()):
        comp_mob = fn_components.get(x)
        fn_comp = MFnSingleIndexedComponent(comp_mob)
        component_ids.extend(fn_comp.getElements())
    return component_ids


def set_component_list_indices(plug: MPlug, indices: list[int]) -> None:
    fn_data: MFnComponentListData = MFnComponentListData()
    data_mob: MObject = fn_data.create()
    fn_comp: MFnSingleIndexedComponent = MFnSingleIndexedComponent()
    comp_mob: MObject = fn_comp.create(MFn.kMeshVertComponent)
    fn_comp.addElements(indices)
    fn_data.add(comp_mob)
    plug.setMObject(data_mob)


def set_point_array(plug: MPlug, point_array: MPointArray) -> None:
    fn_points = MFnPointArrayData()
    points_mob = fn_points.create(point_array)
    plug.setMObject(points_mob)
