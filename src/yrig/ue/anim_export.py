from pathlib import Path

from maya import cmds

import re

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
    export_start: int | None = None,
) -> None:
    """
    Export an animated skeleton to FBX.

    export_start:
        Starting frame of the exported animation.
        None preserves the original frame range.
    """

    start, end = frame_range

    nodes = [skel]
    if export_model:
        nodes.append(model)

    if export_start is not None:
        frame_offset = export_start - start

        if frame_offset != 0:
            joints = [skel] + (
                cmds.listRelatives(
                    skel,
                    allDescendents=True,
                    type="joint",
                    fullPath=True,
                )
                or []
            )

            # Bake animation onto the export skeleton.
            cmds.bakeResults(
                joints,  # type:ignore
                time=(start, end),
                simulation=True,
                sampleBy=1,
                disableImplicitControl=True,
                preserveOutsideKeys=False,
                sparseAnimCurveBake=False,
                minimizeRotation=True,
            )

            # Remove constraints from the export skeleton.
            constraints = (
                cmds.listRelatives(
                    joints,  # type:ignore
                    children=True,
                    type="constraint",
                    fullPath=True,
                )
                or []
            )

            if constraints:
                cmds.delete(constraints)  # type:ignore

            # Shift the baked animation.
            cmds.keyframe(
                joints,  # type:ignore
                animation="objects",
                relative=True,
                timeChange=frame_offset,
            )

            start += frame_offset
            end += frame_offset

    export_fbx_file(
        filepath=path,
        nodes=nodes,
        animation=True,
        bake_animation=True,
        bake_start=start,
        bake_end=end,
        bake_step=1,
    )


def get_namespaces() -> list[str]:
    """Returns all user-created namespaces in the scene."""
    namespaces = (
        cmds.namespaceInfo(
            ":",
            listOnlyNamespaces=True,
            recurse=True,
        )
        or []
    )

    return [namespace for namespace in namespaces if namespace not in {"UI", "shared"}]


def get_rig_namespaces(rigs: list[str]) -> list[str]:
    """Returns namespaces matching rig names, including numbered instances."""
    namespaces = get_namespaces()

    return [
        namespace
        for namespace in namespaces
        if any(re.fullmatch(rf"{re.escape(rig)}\d*", namespace) for rig in rigs)
    ]
