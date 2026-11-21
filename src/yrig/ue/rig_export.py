from pathlib import Path

from maya import cmds

from yrig.io.fbx import export_fbx_file


def publish_ue_rig(
    path: Path,
    model: str = "geo",
    skel: str = "root_C0_root_jnt",
) -> None:
    """
    Exports the rig, with a clean hierarchy, to a given path

    path: the path
    model: the top model transfrom
    skel: the top/ root joint
    """

    model_parent = cmds.listRelatives(model, parent=True, fullPath=True)[0]
    skel_parent = cmds.listRelatives(skel, parent=True, fullPath=True)[0]
    cmds.parent(model, skel, world=True)
    export_fbx_file(filepath=path, nodes=[model, skel])

    cmds.parent(model, model_parent)
    cmds.parent(skel, skel_parent)
