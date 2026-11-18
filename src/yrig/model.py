import re

from maya import cmds

from yrig.name import get_short_name, normalize_name

MODEL_SUFFIX_REGEX = re.compile(r"(?:_(?:mdl|geo|model|mesh))+$", re.IGNORECASE)
SPECIAL_SUFFIX_RE = re.compile(r"_(?:sim|proxy)$", re.IGNORECASE)


def normalize_model_name(name: str) -> str:
    normalized_name = normalize_name(name)
    normalized_name = re.sub(MODEL_SUFFIX_REGEX, "", normalized_name)
    return normalized_name


def add_model_suffix(name: str, suffix: str = "geo") -> str:
    return name if SPECIAL_SUFFIX_RE.search(name) else f"{name}_{suffix}"


def group_top_level_nodes(group_name: str = "geo", nodes: list[str] | None = None) -> str:
    top_level_nodes: list[str]
    if nodes is not None:
        top_level_nodes = cmds.ls(nodes, assemblies=True)  # type: ignore
    else:
        top_level_nodes = cmds.ls(assemblies=True)

    if not top_level_nodes:
        raise RuntimeError("No transforms found in imported model!")

    # Case 1: exactly one root transform (a group)
    if len(top_level_nodes) == 1:
        top_level_node = top_level_nodes[0]
        if cmds.nodeType(top_level_node) == "transform" and not cmds.listRelatives(
            top_level_node, shapes=True
        ):
            geo_grp = top_level_node
        else:
            geo_grp = cmds.group(top_level_node, name=group_name, world=True)
        if geo_grp != group_name:
            geo_grp = cmds.rename(geo_grp, group_name)
    else:
        # Case 2: multiple top level transforms (we need to group them)
        geo_grp = cmds.group(top_level_nodes, name=group_name, world=True)  # type: ignore

    return geo_grp


def merge_group(group: str, name: str | None = None) -> str | None:
    """Replace `group` with a single mesh called `name`."""
    target_name = name or get_short_name(group)
    matches = cmds.ls(group, long=True) or []
    if len(matches) != 1:
        raise ValueError(
            f"Couldn't resolve a path for {group}, got {len(matches)} matches ({matches}): "
        )
    resolved_group = matches[0]

    shapes: list[str] = cmds.listRelatives(
        resolved_group, allDescendents=True, type="mesh", noIntermediate=True, fullPath=True
    )
    if not shapes:
        cmds.delete(resolved_group)
        return None

    meshes = list(dict.fromkeys(cmds.listRelatives(shapes, parent=True, fullPath=True)))  # type: ignore
    if meshes == [resolved_group]:  # group is already a single mesh
        return cmds.rename(resolved_group, target_name)

    parents = cmds.listRelatives(resolved_group, parent=True, fullPath=True)
    parent = parents[0] if parents else None
    sibling_index = cmds.listRelatives(parent or "|", children=True, fullPath=True).index(
        resolved_group
    )
    merged = (
        meshes[0] if len(meshes) == 1 else cmds.polyUnite(meshes, ch=False, mergeUVSets=True)[0]  # type: ignore
    )
    if cmds.listRelatives(merged, parent=True, fullPath=True) != parents:
        if parent:
            cmds.parent(merged, parent, world=not parent)
        else:
            cmds.parent(merged, world=True)

    target_path = f"{parent}|{target_name}" if parent else f"|{target_name}"
    if cmds.objExists(target_path):
        cmds.delete(target_path)
    final_name = cmds.rename(merged, target_name)
    cmds.reorder(final_name, front=True)
    cmds.reorder(final_name, relative=sibling_index)
    return final_name
