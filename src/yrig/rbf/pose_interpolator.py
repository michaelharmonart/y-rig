from collections.abc import Callable, Iterable
from pathlib import Path

from maya import cmds

from yrig.deformer.blendshape import export_blendshape, import_blendshape
from yrig.maya_api.attribute import IntegerAttribute
from yrig.maya_api.node import PoseInterpolatorManager
from yrig.transform import create_transform


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


def group_pose_interpolators_by_directory(
    pose_interpolators: Iterable[str],
    parent: str | None = None,
    group_namer: Callable[[str], str] | None = None,
) -> list[str]:
    manager = PoseInterpolatorManager.from_existing("poseInterpolatorManager")
    created_groups = []
    index_group_map: dict[int, str] = {}

    def get_group_name(directory_index: int) -> str:
        directory_name = manager.pose_interpolator_directory[directory_index].directory_name.get()

        return group_namer(directory_name) if group_namer else f"{directory_name}_rbf"

    for pose_interpolator in pose_interpolators:
        shape = resolve_pose_interpolator_shape(pose_interpolator)
        transform: str | None = next(
            iter(cmds.listRelatives(shape, parent=True, type="transform") or []),
            None,
        )
        destinations = (
            cmds.listConnections(
                shape,
                source=False,
                destination=True,
                plugs=True,
            )
            or []
        )
        if not destinations:
            continue

        directory_index: int = IntegerAttribute(destinations[0]).get()

        # Walk upward only until we find an existing group.
        missing: list[int] = []
        group_parent = parent

        while directory_index != 0:
            if directory_index in index_group_map:
                group_parent = index_group_map[directory_index]
                break

            group_name = get_group_name(directory_index)

            if cmds.objExists(group_name):
                index_group_map[directory_index] = group_name
                group_parent = group_name
                break

            missing.append(directory_index)

            directory_index = manager.pose_interpolator_directory[
                directory_index
            ].parent_index.get()

        # Create the missing part of the hierarchy from top -> bottom.
        for directory_index in reversed(missing):
            group_name = get_group_name(directory_index)

            create_transform(group_name, group_parent)

            index_group_map[directory_index] = group_name
            created_groups.append(group_name)

            group_parent = group_name
        if transform and group_parent:
            cmds.parent(transform, group_parent, relative=True)

    return created_groups


def import_pose_file(
    filepath: Path, parent: str | None = None, import_shapes: bool = True
) -> set[str]:
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
    created_pose_interps = current_pose_interps - existing_pose_interps
    group_pose_interpolators_by_directory(created_pose_interps, parent)

    return created_pose_interps


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
