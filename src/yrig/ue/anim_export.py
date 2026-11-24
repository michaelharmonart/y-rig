import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from maya import cmds, mel

from yrig.io.fbx import export_fbx_file, import_fbx_file

RIG_PUBLISH_DIRECTORY = Path("/groups/lantern/Production/asset/Characters")


@dataclass
class ExportSkeleton:
    root: str
    model: str | None
    joints: list[str]
    imported_nodes: list[str]
    constraints: list[str]


@contextmanager
def temporary_export_skel(
    rig: str,
    namespace: str,
    skel: str = "root_C0_root_jnt",
    model: str = "geo",
) -> Generator[ExportSkeleton, None, None]:
    """Import an export skeleton and clean it up after use."""

    original_nodes = get_scene_node_ids()

    original_selection = (
        cmds.ls(
            selection=True,
            long=True,
        )
        or []
    )

    original_time = cmds.currentTime(query=True)

    try:
        yield load_export_skel(
            rig=rig,
            namespace=namespace,
            skel=skel,
            model=model,
        )

    finally:
        delete_new_scene_nodes(original_nodes)

        cmds.currentTime(original_time)

        selection = [node for node in original_selection if cmds.objExists(node)]

        if selection:
            cmds.select(selection, replace=True)  # type:ignore
        else:
            cmds.select(clear=True)


def reset_fbx_preferences() -> None:
    """Restore Maya's default FBX settings after publishing."""

    mel.eval("FBXResetImport;")
    mel.eval("FBXResetExport;")

    mel.eval("FBXProperty Import|IncludeGrp|Animation -v true;")
    mel.eval("FBXImportSetTake -ti -1;")
    mel.eval("FBXImportFillTimeline -v true;")
    mel.eval('FBXImportMode -v "add";')


def get_rig_publish_path(rig: str) -> Path:
    """Resolve the published FBX for a rig asset."""
    return RIG_PUBLISH_DIRECTORY / rig / "publish" / "rig" / f"{rig}.fbx"


def resolve_rig_name(
    namespace: str,
    rigs: list[str],
) -> str:
    """Resolve an asset name from its rig instance namespace."""

    namespace = namespace.strip(":")

    if namespace in rigs:
        return namespace

    for rig in sorted(rigs, key=len, reverse=True):
        if re.fullmatch(rf"{re.escape(rig)}\d*", namespace):
            return rig

    raise ValueError(f"Could not resolve rig asset from namespace: {namespace}")


def load_export_skel(
    rig: str,
    namespace: str | None = None,
    skel: str = "root_C0_root_jnt",
    model: str = "geo",
) -> ExportSkeleton:
    """Import a clean skeleton and constrain it to a rig instance."""

    if namespace is None:
        namespace = rig

    rig_publish_path = get_rig_publish_path(rig)

    if not rig_publish_path.is_file():
        raise FileNotFoundError(rig_publish_path)

    # Avoid name collisions with existing export skeletons.
    if cmds.objExists(f":{skel}"):
        raise RuntimeError(f"An unnamespaced skeleton already exists: {skel}")

    previous_namespace = cmds.namespaceInfo(
        currentNamespace=True,
        absoluteName=True,
    )

    # Import into Maya's root namespace.
    try:
        cmds.namespace(setNamespace=":")

        imported_nodes = (
            import_fbx_file(
                filepath=rig_publish_path,
            )
            or []
        )

    finally:
        cmds.namespace(setNamespace=previous_namespace)

    imported_joints = (
        cmds.ls(
            imported_nodes,  # type:ignore
            type="joint",
            long=True,
        )
        or []
    )

    roots = [joint for joint in imported_joints if joint.split("|")[-1] == skel]

    if len(roots) != 1:
        raise RuntimeError(f"Could not uniquely identify clean export skeleton: {skel}")

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

    imported_transforms = (
        cmds.ls(
            imported_nodes,  # type:ignore
            type="transform",
            long=True,
        )
        or []
    )

    models = [node for node in imported_transforms if node.split("|")[-1] == model]

    model_node = models[0] if len(models) == 1 else None

    constraints = constrain_export_skel(
        joints=joints,
        namespace=namespace,
    )

    return ExportSkeleton(
        root=root,
        model=model_node,
        joints=joints,
        imported_nodes=imported_nodes,
        constraints=constraints,
    )


def report_skeleton_warning(message: str) -> None:
    cmds.warning(message)

    if not cmds.about(batch=True):
        cmds.confirmDialog(
            title="Export Skeleton Warning",
            message=message,
            button=["OK"],
            defaultButton="OK",
            icon="warning",
        )


def constrain_export_skel(
    joints: list[str],
    namespace: str,
) -> list[str]:

    constraints = []

    missing_joints = []
    failed_constraints = []

    for joint in joints:
        joint_name = joint.split("|")[-1].split(":")[-1]
        driver = f"{namespace}:{joint_name}"

        matches = (
            cmds.ls(
                driver,
                type="joint",
                long=True,
            )
            or []
        )

        if len(matches) != 1:
            missing_joints.append(driver)
            continue

        driver = matches[0]

        try:
            constraints.extend(
                cmds.parentConstraint(  # type:ignore
                    driver,
                    joint,
                    maintainOffset=False,
                )
            )

            constraints.extend(
                cmds.scaleConstraint(  # type:ignore
                    driver,
                    joint,
                    maintainOffset=False,
                )
            )

        except RuntimeError as exc:
            failed_constraints.append(f"{joint_name}: {exc}")

    messages = []

    if missing_joints:
        messages.append("Missing or ambiguous rig joints:\n" + "\n".join(missing_joints))

    if failed_constraints:
        messages.append("Failed constraints:\n" + "\n".join(failed_constraints))

    if messages:
        report_skeleton_warning("\n\n".join(messages))

    return constraints


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


def get_scene_node_ids() -> set[str]:
    """Returns UUIDs of all dependency nodes in the scene."""
    return set(cmds.ls(dependencyNodes=True, uuid=True) or [])


def delete_new_scene_nodes(
    original_nodes: set[str],
) -> None:
    """Delete nodes created since the original scene snapshot."""

    new_nodes = get_scene_node_ids() - original_nodes

    for node_id in new_nodes:
        matches = cmds.ls(node_id, long=True) or []

        if not matches:
            continue

        try:
            cmds.delete(matches[0])

        except RuntimeError as exc:
            cmds.warning(f"Could not delete temporary node {matches[0]}: {exc}")


def publish_rig_animation(
    path: Path,
    rig: str,
    namespace: str,
    frame_range: tuple[int, int],
    export_model: bool = False,
    export_start: int | None = 0,
) -> None:
    """Publish animation using a temporary clean export skeleton."""

    try:
        with temporary_export_skel(
            rig=rig,
            namespace=namespace,
        ) as skeleton:
            if export_model and skeleton.model is None:
                raise RuntimeError("Could not find the imported export model.")

            publish_ue_anim(
                path=path,
                frame_range=frame_range,
                skel=skeleton.root,
                model=skeleton.model or "geo",
                export_model=export_model,
                export_start=export_start,
            )

    finally:
        try:
            reset_fbx_preferences()

        except RuntimeError as exc:
            cmds.warning(f"Failed to reset FBX preferences: {exc}")
