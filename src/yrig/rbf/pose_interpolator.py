from dataclasses import dataclass
from pathlib import Path
from typing import cast

from maya import cmds, mel


@dataclass
class PoseInterpolatorDriver:
    name: str
    index: int
    matrix_attr: str


@dataclass
class PoseInterpolatorPose:
    name: str
    index: int
    attr: str


@dataclass
class PoseInterpolator:
    node: str
    drivers: list[PoseInterpolatorDriver]
    poses: list[PoseInterpolatorPose]

    def get_pose(self, pose: str | int) -> PoseInterpolatorPose:
        """
        Get a pose using either its name or logical index.
        """

        for pose_data in self.poses:
            if pose_data.name == pose:
                return pose_data

            if pose_data.index == pose:
                return pose_data

        raise ValueError(f"Pose {pose!r} does not exist on {self.node}")

    def get_driver(
        self,
        driver: str | int,
    ) -> PoseInterpolatorDriver:
        """
        Get a driver using either its name or logical index.
        """

        for driver_data in self.drivers:
            if driver_data.name == driver:
                return driver_data

            if driver_data.index == driver:
                return driver_data

        raise ValueError(f"Driver {driver!r} does not exist on {self.node}")


def import_pose_interpolator(
    path: str | Path,
    pose_interp_parent: str = "",
) -> list[PoseInterpolator]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pose Interpolator file does not exist: {path}")

    existing_pose_interps = set(cmds.ls(type="poseInterpolator") or [])

    try:
        mel.eval(f'poseInterpolatorImportPoses "{path.as_posix()}" 1;')

    except Exception as error:
        raise RuntimeError(f"Failed to import Pose Interpolator file: {path}") from error

    current_pose_interps = set(cmds.ls(type="poseInterpolator") or [])

    created_pose_interps = sorted(current_pose_interps - existing_pose_interps)

    if pose_interp_parent and created_pose_interps:
        if not cmds.objExists(pose_interp_parent):
            raise RuntimeError(f"Pose Interpolator parent does not exist: {pose_interp_parent}")

        cmds.parent(
            created_pose_interps,  # type:ignore
            pose_interp_parent,
        )

    return [get_pose_interpolator_data(node) for node in created_pose_interps]


def export_pose_interpolator(
    path: Path,
    pose_interp: str,
) -> Path:
    """
    Export a Pose Interpolator node.

    Args:
        path:
            Output path.

        pose_interp:
            Pose Interpolator node to export.

    Returns:
        The exported path.
    """

    validate_pose_interpolator(pose_interp)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose_path = path.as_posix()

    try:
        mel.eval(
            f'''
            string $tpls[] = {{"{pose_interp}"}};
            string $poses[] = {{}};

            poseInterpolatorExportPoses(
                "{pose_path}",
                $tpls,
                $poses,
                1
            );
            '''
        )

    except Exception as error:
        raise RuntimeError(f"Failed to export Pose Interpolator: {pose_interp}") from error

    return path


def get_pose_interpolator_data(
    pose_interp_node: str,
) -> PoseInterpolator:
    """
    Read a Pose Interpolator node into a data class.

    Args:
        pose_interp_node:
            Pose Interpolator node to read.

    Returns:
        Structured Pose Interpolator information.
    """

    validate_pose_interpolator(pose_interp_node)

    drivers = get_pose_interpolator_drivers(pose_interp_node)

    poses = get_pose_interpolator_poses(pose_interp_node)

    return PoseInterpolator(
        node=pose_interp_node,
        drivers=drivers,
        poses=poses,
    )


def get_pose_interpolator_drivers(
    pose_interp_node: str,
) -> list[PoseInterpolatorDriver]:
    """
    Get the drivers used by a Pose Interpolator.
    """

    driver_names = (
        cast(
            list[str] | None,
            cmds.poseInterpolator(
                pose_interp_node,
                query=True,
                drivers=True,
            ),
        )
        or []
    )

    driver_indices = get_multi_indices(f"{pose_interp_node}.driver")

    drivers: list[PoseInterpolatorDriver] = []

    for list_index, driver_name in enumerate(driver_names):
        if list_index < len(driver_indices):
            driver_index = driver_indices[list_index]
        else:
            driver_index = list_index

        drivers.append(
            PoseInterpolatorDriver(
                name=driver_name,
                index=driver_index,
                matrix_attr=(f"{pose_interp_node}.driver[{driver_index}].driverMatrix"),
            )
        )

    return drivers


def get_pose_interpolator_poses(
    pose_interp_node: str,
) -> list[PoseInterpolatorPose]:
    """
    Get the poses and output attributes from a Pose Interpolator.
    """

    pose_names = (
        cast(
            list[str] | None,
            cmds.poseInterpolator(
                pose_interp_node,
                query=True,
                poseNames=True,
            ),
        )
        or []
    )

    pose_indices = get_multi_indices(f"{pose_interp_node}.pose")

    poses: list[PoseInterpolatorPose] = []

    for list_index, pose_name in enumerate(pose_names):
        if list_index < len(pose_indices):
            pose_index = pose_indices[list_index]
        else:
            pose_index = list_index

        poses.append(
            PoseInterpolatorPose(
                name=pose_name,
                index=pose_index,
                attr=(f"{pose_interp_node}.output[{pose_index}]"),
            )
        )

    return poses


def get_multi_indices(attribute: str) -> list[int]:
    """
    Return the populated logical indices of a Maya multi attribute.
    """

    if not cmds.objExists(attribute):
        return []

    indices = cmds.getAttr(
        attribute,
        multiIndices=True,
    )

    if not indices:
        return []

    return sorted(cast(list[int], indices))


def validate_pose_interpolator(
    pose_interp_node: str,
) -> None:
    """
    Validate that a node exists and is a Pose Interpolator.
    """

    if not cmds.objExists(pose_interp_node):
        raise RuntimeError(f"Pose Interpolator does not exist: {pose_interp_node}")

    if cmds.nodeType(pose_interp_node) != "poseInterpolator":
        raise TypeError(f"{pose_interp_node} is not a poseInterpolator node")
