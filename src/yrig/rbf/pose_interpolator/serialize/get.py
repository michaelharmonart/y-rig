from collections.abc import Iterable

from yrig.maya_api.node import PoseInterpolatorManagerNode, PoseInterpolatorNode
from yrig.rbf.pose_interpolator.serialize.directory import (
    get_directory_indices,
    get_pose_interpolator_indices,
)

from .data import (
    PoseInterpolatorControllerDataItem,
    PoseInterpolatorData,
    PoseInterpolatorDirectoryData,
    PoseInterpolatorDriverData,
    PoseInterpolatorPoseControllerData,
    PoseInterpolatorPoseData,
)


def _str_or_none(value: object | None) -> str | None:
    return None if value is None else str(value)


def get_pose_interpolator_data(
    pose_interpolator: str | PoseInterpolatorNode,
) -> PoseInterpolatorData:
    node = (
        pose_interpolator
        if isinstance(pose_interpolator, PoseInterpolatorNode)
        else PoseInterpolatorNode.from_existing(pose_interpolator)
    )
    enable_rotation = node.enable_rotation.get()
    enable_translation = node.enable_translation.get()

    drivers = []
    for index in node.driver.get_indices():
        driver = node.driver[index]
        controllers = {
            controller_index: str(driver.driver_controller[controller_index].get_input())
            for controller_index in driver.driver_controller.get_indices()
        }
        drivers.append(
            PoseInterpolatorDriverData(
                matrix=str(driver.driver_matrix.get_input()),
                orient=driver.driver_orient.get(),
                rotate_axis=driver.driver_rotate_axis.get(),
                twist_axis=driver.driver_twist_axis.get(),
                rotate_order=driver.driver_rotate_order.get(),
                euler_twist=driver.driver_euler_twist.get(),
                controllers=controllers,
            )
        )

    poses = []
    for index in node.pose.get_indices():
        pose = node.pose[index]

        controller_data = {}
        for controller_index in pose.pose_controller_data.get_indices():
            controller = pose.pose_controller_data[controller_index]

            items = {}
            for item_index in controller.pose_controller_data_item.get_indices():
                item = controller.pose_controller_data_item[item_index]
                items[item_index] = PoseInterpolatorControllerDataItem(
                    name=item.pose_controller_data_item_name.get(),
                    type=item.pose_controller_data_item_type.get(),
                    value=item.pose_controller_data_item_value.get(),
                )

            controller_data[controller_index] = PoseInterpolatorPoseControllerData(
                items=items,
            )

        rotations: list[tuple[float, float, float, float]] = (  # type: ignore
            [tuple(pose.pose_rotation[index].get()) for index in pose.pose_rotation.get_indices()]
            if enable_rotation
            else None
        )

        translations: list[tuple[float, float, float]] = (  # type: ignore
            [
                tuple(pose.pose_translation[index].get())
                for index in pose.pose_translation.get_indices()
            ]
            if enable_translation
            else None
        )

        poses.append(
            PoseInterpolatorPoseData(
                name=pose.pose_name.get(),
                rotations=rotations,
                translations=translations,
                independent=pose.is_independent.get(),
                rotation_falloff=pose.pose_rotation_falloff.get(),
                translation_falloff=pose.pose_translation_falloff.get(),
                pose_type=pose.pose_type.get(),
                gaussian_falloff=pose.pose_falloff.get(),
                is_enabled=pose.is_enabled.get(),
                controller_data=controller_data,
            )
        )

    outputs = [[str(attr) for attr in output.get_outputs()] for output in node.output]
    return PoseInterpolatorData(
        name=node.name,
        allow_negative_weights=node.allow_negative_weights.get(),
        enable_rotation=enable_rotation,
        enable_translation=enable_translation,
        interpolation=node.interpolation.get(),
        output_smoothing=node.output_smoothing.get(),
        regularization=node.regularization.get(),
        drivers=drivers,
        poses=poses,
        outputs=outputs,
    )


def _get_pose_interpolator_directory_data(
    manager: PoseInterpolatorManagerNode,
    index: int,
    directory_indices: set[int],
    pose_interpolator_indices: set[int],
    export_all: bool,
    ancestor_selected: bool = False,
) -> PoseInterpolatorDirectoryData | None:
    """
    Collect pose-interpolator data from a directory.

    A directory is retained when:
    - It was explicitly selected.
    - One of its ancestors was selected.
    - It contains a selected child directory.
    - It contains a selected pose interpolator.
    """
    directory = manager.pose_interpolator_directory[index]

    directory_selected = export_all or ancestor_selected or index in directory_indices

    child_indices = directory.child_indices.get()
    child_directories = []
    child_pose_interpolators = []
    for child_index in child_indices:
        if child_index < 0:
            child_directory_data = _get_pose_interpolator_directory_data(
                manager,
                -child_index,
                directory_indices,
                pose_interpolator_indices,
                export_all,
                ancestor_selected=directory_selected,
            )
            if child_directory_data is not None:
                child_directories.append(child_directory_data)
        else:
            if directory_selected or child_index in pose_interpolator_indices:
                parent_attr = manager.pose_interpolator_parent[child_index]
                source_attr = parent_attr.get_input()
                if source_attr is None:
                    raise RuntimeError(
                        f"Couldn't find a poseInterpolator connected to {source_attr}"
                    )
                source_node = str(source_attr).split(".", 1)[0]
                child_pose_interpolators.append(get_pose_interpolator_data(source_node))

    should_keep_directory = (
        index == 0 or directory_selected or child_directories or child_pose_interpolators
    )

    if not should_keep_directory:
        return None

    return PoseInterpolatorDirectoryData(
        name=directory.directory_name.get(),
        directories=child_directories,
        pose_interpolators=child_pose_interpolators,
    )


def get_pose_interpolator_directory_data(
    directories: Iterable[str] | None = None,
    pose_interpolators: Iterable[str] | None = None,
) -> PoseInterpolatorDirectoryData:
    """
    Get serialized pose-interpolator directory data.

    Args:
        index: Manager directory index to begin exporting from.
        directories: Directory names to export. Every pose interpolator and
            nested directory below a selected directory is exported.
        pose_interpolators: Pose-interpolator names to export.

    Ancestor directories are automatically retained for selected pose
    interpolators and nested selected directories.
    """
    manager = PoseInterpolatorManagerNode.from_existing("poseInterpolatorManager")
    export_all = directories is None and pose_interpolators is None

    if export_all:
        directory_indices: set[int] = set()
        pose_interpolator_indices: set[int] = set()
    else:
        directory_names = set(directories or ())
        pose_interpolator_names = set(pose_interpolators or ())

        directory_indices = set(get_directory_indices(manager, directory_names).values())
        pose_interpolator_indices = set(
            get_pose_interpolator_indices(manager, pose_interpolator_names).values()
        )

    directory_data = _get_pose_interpolator_directory_data(
        manager=manager,
        index=0,
        directory_indices=directory_indices,
        pose_interpolator_indices=pose_interpolator_indices,
        export_all=export_all,
    )

    if directory_data is None:
        raise RuntimeError(f"Couldn't create pose-interpolator directory data for index {0}")

    return directory_data
