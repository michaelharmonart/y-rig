from pathlib import Path

from maya import cmds

from yrig.deformer.blendshape import export_blendshape, import_blendshape


def resolve_pose_interpolator_shape(pose_interpolator: str) -> str:
    if cmds.nodeType(pose_interpolator) == "poseInterpolator":
        return pose_interpolator
    shapes = cmds.listRelatives(
        pose_interpolator,
        shapes=True,
        type="poseInterpolator",
    )
    if not shapes:
        raise RuntimeError(f"Couldn't find a shape for {pose_interpolator}")
    return shapes[0]


def get_pose_index(pose_interpolator_shape: str, pose_name: str) -> int:
    attr = f"{pose_interpolator_shape}.pose"

    for index in cmds.getAttr(attr, multiIndices=True) or []:
        name = cmds.getAttr(f"{attr}[{index}].poseName")
        if name == pose_name:
            return index

    raise RuntimeError(f"Couldn't resolve and index for the pose {pose_name}")


def _reslove_pose_index(pose_interpolator: str, pose: str | int) -> int:
    shape = resolve_pose_interpolator_shape(pose_interpolator)
    return pose if isinstance(pose, int) else get_pose_index(shape, pose)


def get_pose_interpolator_blendshapes(pose_interpolator: str) -> list[str]:
    shape = resolve_pose_interpolator_shape(pose_interpolator)

    blendshapes: list[str] = []
    for index in cmds.poseInterpolator(shape, query=True, index=True) or []:
        plug = f"{shape}.output[{index}]"

        for node in cmds.listConnections(plug, source=False) or []:
            if cmds.nodeType(node) == "blendShape":
                blendshapes.append(node)

    return list(blendshapes)


def import_pose_file(filepath: Path, import_shapes: bool = True) -> set[str]:
    if filepath.suffix != ".pose":
        raise ValueError(f"The file at {filepath} is not a .pose file.")
    if not filepath.exists():
        raise FileNotFoundError(f"No .pose file found at {filepath}.")

    if import_shapes:
        for shp_file in filepath.parent.glob(f"{filepath.stem}.*.shp"):
            blendshape = shp_file.stem.removeprefix(f"{filepath.stem}.")
            import_blendshape(shp_file, blendshape)

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
    pose_interpolators: list[str] | None = None,
    poses: list[tuple[str, str | int]] | None = None,
    export_shapes: bool = True,
) -> None:
    if pose_interpolators:
        resloved_pose_interpolators = [
            resolve_pose_interpolator_shape(pose_interpolator)
            for pose_interpolator in pose_interpolators
        ]
    elif poses:
        resloved_pose_interpolators = [
            resolve_pose_interpolator_shape(pose_interpolator) for pose_interpolator, _pose in poses
        ]
    else:
        raise ValueError("Must give pose_interpolators or poses for export")

    _validate_pose_interpolators(resloved_pose_interpolators)

    args = []
    if pose_interpolators:
        args.append(pose_interpolators)
    kwargs = {}
    if poses:
        kwargs["pose"] = poses

    cmds.poseInterpolator(
        *args,  # type: ignore
        **kwargs,  # type: ignore
        edit=True,
        exportPoses=str(filepath),
    )

    if not export_shapes:
        return

    # Find connected shape deformers.
    blendshapes: list[str] = []

    for pose_interpolator in resloved_pose_interpolators:
        blendshapes.extend(get_pose_interpolator_blendshapes(pose_interpolator))

    for blendshape in blendshapes:
        shape_file = filepath.with_suffix(f".{blendshape}.shp")
        targets: list[str] = []
        destinations: list[str] = []
        if poses:
            for pose_interpolator, pose in poses:
                pose_index = _reslove_pose_index(pose_interpolator, pose)
                source = (
                    f"{resolve_pose_interpolator_shape(pose_interpolator)}.output[{pose_index}]"
                )
                destinations.extend(
                    cmds.connectionInfo(  # type: ignore
                        source,
                        destinationFromSource=True,
                    )
                    or []
                )
        else:
            for pose_interpolator in resloved_pose_interpolators:
                target_name = f"{pose_interpolator}.output"
                indices = cmds.getAttr(target_name, multiIndices=True) or []

                for index in indices:
                    destinations.extend(
                        cmds.connectionInfo(  # type: ignore
                            f"{target_name}[{index}]",
                            destinationFromSource=True,
                        )
                        or []
                    )

        for dest in destinations:
            node, target_name = dest.split(".", 1)
            if node == blendshape:
                targets.append(target_name)

        export_blendshape(
            shape_file,
            blendshape,
            targets,
        )
