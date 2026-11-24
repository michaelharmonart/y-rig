from pathlib import Path

from maya import cmds

from yrig.io.fbx import export_fbx_file, import_fbx_file


def load_export_skel(
    rig: str,
    skel: str = "root_C0_root_jnt",
) -> list | None:
    """Import an export skeleton and constrain it to the animated rig.
    Returns all imported nodes
    """

    rig_publish_path = (
        Path("/groups/lantern/Production/asset/Characters") / rig / "publish" / "rig" / f"{rig}.fbx"
    )

    imported_nodes = import_fbx_file(filepath=rig_publish_path)

    imported_joints = (
        cmds.ls(
            imported_nodes,  # type:ignore
            type="joint",
            long=True,
        )
        or []
    )

    # Find the root joint of the newly imported skeleton.
    roots = [joint for joint in imported_joints if joint.split("|")[-1].split(":")[-1] == skel]

    if len(roots) != 1:
        cmds.warning(f"Could not uniquely identify export skeleton: {skel}")
        return

    root = roots[0]

    joints = [root] + (
        cmds.listRelatives(
            root,
            allDescendents=True,
            type="joint",
            fullPath=True,
        )
        or []
    )

    missing_joints = []
    failed_constraints = []

    for joint in joints:
        joint_name = joint.split("|")[-1].split(":")[-1]
        driver = f"{rig}:{joint_name}"

        matches = cmds.ls(driver, type="joint", long=True) or []

        if len(matches) != 1:
            missing_joints.append(driver)
            continue

        driver = matches[0]

        try:
            cmds.parentConstraint(
                driver,
                joint,
                maintainOffset=False,
            )

            cmds.scaleConstraint(
                driver,
                joint,
                maintainOffset=False,
            )

        except RuntimeError as exc:
            failed_constraints.append(f"{joint_name}: {exc}")

    # Report mismatches without interrupting the setup.
    messages = []

    if missing_joints:
        messages.append("Missing or ambiguous rig joints:\n" + "\n".join(missing_joints))

    if failed_constraints:
        messages.append("Failed constraints:\n" + "\n".join(failed_constraints))

    if messages:
        message = "\n\n".join(messages)

        cmds.warning(message)

        if not cmds.about(batch=True):
            cmds.confirmDialog(
                title="Export Skeleton Warning",
                message=message,
                button=["OK"],
                defaultButton="OK",
                icon="warning",
            )
    return imported_nodes


def publish_ue_anim(
    path: Path,
    frame_range: tuple,
    model: str = "geo",
    skel: str = "root_C0_root_jnt",
    export_model: bool = False,
) -> None:
    """
    Exports the rig, with a clean hierarchy, to a given path

    path: the path
    model: the top model transfrom
    skel: the top/ root joint
    """

    if export_model:
        # cmds.parent(model, skel, world=True)
        export_fbx_file(
            bake_animation=True,
            filepath=path,
            nodes=[skel, model],
            animation=True,
            bake_start=frame_range[0],
            bake_end=frame_range[1],
            bake_step=1,
        )

        # cmds.parent(model, model_parent)
    else:
        export_fbx_file(
            bake_animation=True,
            filepath=path,
            nodes=[skel],
            animation=True,
            bake_start=frame_range[0],
            bake_end=frame_range[1],
            bake_step=1,
        )
    # cmds.parent(skel, skel_parent)
