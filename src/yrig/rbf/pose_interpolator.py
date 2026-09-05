from pathlib import Path

from maya import cmds

from yrig.deformer.blendshape import export_blendshape
from yrig.shape import get_shape


def resolve_pose_interpolator_shape(pose_interpolator: str) -> str:
    shape = get_shape(pose_interpolator)
    if shape is None:
        raise RuntimeError(f"Couldn't find a shape for {pose_interpolator}")
    return shape


def pose_interpolator_pose_index(pose_interpolator_shape: str, pose_name: str) -> int:
    attr = f"{pose_interpolator_shape}.pose"

    for index in cmds.getAttr(attr, multiIndices=True) or []:
        name = cmds.getAttr(f"{attr}[{index}].poseName")
        if name == pose_name:
            return index

    raise RuntimeError(f"Couldn't resolve and index for the pose {pose_name}")


def reslove_pose_interpolator_pose_index(pose_interpolator: str, pose: str | int) -> int:
    shape = resolve_pose_interpolator_shape(pose_interpolator)
    return pose if isinstance(pose, int) else pose_interpolator_pose_index(shape, pose)


def pose_interpolator_connected_shape_deformers(pose_interpolator: str) -> list[str]:
    shape = resolve_pose_interpolator_shape(pose_interpolator)

    deformers: list[str] = []
    for index in cmds.poseInterpolator(shape, query=True, index=True) or []:
        plug = f"{shape}.output[{index}]"

        for node in cmds.listConnections(plug, source=False) or []:
            if cmds.nodeType(node) == "blendShape":
                deformers.append(node)

    return list(deformers)


def import_pose_file(filepath: Path, import_shapes: bool = True) -> set[str]:
    if filepath.suffix != ".pose":
        raise ValueError(f"The file at {filepath} is not a .pose file.")
    if not filepath.exists():
        raise FileNotFoundError(f"No .pose file found at {filepath}.")

    if import_shapes:
        for shp_file in filepath.parent.glob(f"{filepath.stem}.*.shp"):
            bs_name = shp_file.stem.removeprefix(f"{filepath.stem}.")
            if cmds.objExists(bs_name):
                cmds.blendShape(bs_name, edit=True, ip=str(shp_file))
                continue
            else:
                cmds.blendShape(name=bs_name, ip=str(shp_file))

    existing_pose_interps = set(cmds.ls(type="poseInterpolator") or [])
    cmds.poseInterpolator(importPoses=str(filepath))
    current_pose_interps = set(cmds.ls(type="poseInterpolator") or [])

    return current_pose_interps - existing_pose_interps


def _validate_pose_interpolators(
    pose_interpolators: list[str],
) -> None:
    for pose_interpolator in pose_interpolators:
        if (
            not cmds.objExists(pose_interpolator)
            or cmds.nodeType(pose_interpolator) != "poseInterpolator"
        ):
            raise RuntimeError(f"Not a pose interpolator: {pose_interpolator}")


def export_pose_file(
    filepath: Path,
    pose_interpolators: list[str] | None,
    poses: list[tuple[str, str | int]] | None,
    export_shapes: bool = True,
) -> None:
    if pose_interpolators:
        resloved_pose_interpolators = pose_interpolators
    elif poses:
        resloved_pose_interpolators = [pose_intepolator for pose_intepolator, pose in poses]
    else:
        raise ValueError("Must give pose_interpolators or poses for export")

    _validate_pose_interpolators(resloved_pose_interpolators)

    kwargs = {}
    if poses:
        kwargs["pose"] = poses

    cmds.poseInterpolator(
        *kwargs,
        edit=True,
        exportPoses=str(filepath),
    )

    if not export_shapes:
        return

    # Find connected shape deformers.
    blend_shape_deformers: list[str] = []

    for pose_interpolator in resloved_pose_interpolators:
        blend_shape_deformers.extend(pose_interpolator_connected_shape_deformers(pose_interpolator))

    for blendshape in blend_shape_deformers:
        bs_file = filepath.with_suffix(f".{blendshape}.shp")
        targets: list[str] = []

        if poses:
            for pose_interpolator, pose in poses:
                pose_index = reslove_pose_interpolator_pose_index(pose_interpolator, pose)
                source = (
                    f"{resolve_pose_interpolator_shape(pose_interpolator)}.output[{pose_index}]"
                )
                destinations: list[str] = (  # type: ignore
                    cmds.connectionInfo(
                        source,
                        destinationFromSource=True,
                    )
                    or []
                )
        else:
            for pose_interpolator in resloved_pose_interpolators:
                attr = f"{pose_interpolator}.output"
                indices = cmds.getAttr(attr, multiIndices=True) or []
                for index in indices:
                    destinations: list[str] = (  # type: ignore
                        cmds.connectionInfo(
                            f"{attr}[{index}]",
                            destinationFromSource=True,
                        )
                        or []
                    )

        for dest in destinations:
            node, attr = dest.split(".", 1)
            if node == blendshape:
                targets.append(attr)

        export_blendshape(
            blendshape,
            bs_file,
            targets,
        )
