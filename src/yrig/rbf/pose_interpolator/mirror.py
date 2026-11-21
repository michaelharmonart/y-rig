from collections.abc import Callable, Iterable

from maya import cmds

from yrig.maya_api.node import PoseInterpolatorManagerNode, PoseInterpolatorNode
from yrig.name import flip_side
from yrig.rbf.pose_interpolator.core import create_pose_interpolator, resolve_pose_interpolator_node
from yrig.rbf.pose_interpolator.serialize import (
    apply_pose_interpolator_data,
    get_pose_interpolator_data,
)
from yrig.rbf.pose_interpolator.serialize.data import (
    PoseInterpolatorControllerDataItem,
    PoseInterpolatorData,
    PoseInterpolatorDriverData,
    PoseInterpolatorPoseControllerData,
    PoseInterpolatorPoseData,
)


def mirror_pose_interpolator(
    source: PoseInterpolatorNode,
    destination: PoseInterpolatorNode,
    map_mirror: Callable[[str], str] | None = None,
) -> None:
    mirror_func = flip_side if map_mirror is None else map_mirror
    source_data = get_pose_interpolator_data(source)

    mirrored_name = mirror_func(source_data.name)

    mirrored_drivers = []
    for source_driver in source_data.drivers:
        mirrored_controllers = {}
        for index, source_controller in source_driver.controllers.items():
            mirrored_controller = mirror_func(source_controller)
            if not cmds.objExists(mirrored_controller):
                raise RuntimeError(
                    f"The mirrored control attribute {mirrored_controller} doesn't exist!"
                )
            mirrored_controllers[index] = mirrored_controller

        mirrored_matrix = mirror_func(source_driver.matrix)
        if not cmds.objExists(mirrored_matrix):
            raise RuntimeError(f"The mirrored matrix attribute {mirrored_matrix} doesn't exist!")
        mirrored_driver = PoseInterpolatorDriverData(
            matrix=mirrored_matrix,
            orient=source_driver.orient,
            rotate_axis=source_driver.rotate_axis,
            twist_axis=source_driver.twist_axis,
            rotate_order=source_driver.rotate_order,
            euler_twist=source_driver.euler_twist,
            controllers=mirrored_controllers,
        )
        mirrored_drivers.append(mirrored_driver)

    mirrored_poses = []
    for source_pose in source_data.poses:
        mirrored_controller_data_dict = {}
        for index, source_controller_data in source_pose.controller_data.items():
            mirrored_controller_data_item_dict = {}
            for item_index, source_item in source_controller_data.items.items():
                mirrored_name = mirror_func(source_item.name)
                if not cmds.objExists(mirrored_name):
                    raise RuntimeError(
                        f"The mirrored control attribute {mirrored_name} doesn't exist!"
                    )
                mirrored_controller_data_item_dict[item_index] = PoseInterpolatorControllerDataItem(
                    name=mirrored_name, type=source_item.type, value=source_item.value
                )
            mirrored_controller_data = PoseInterpolatorPoseControllerData(
                mirrored_controller_data_item_dict
            )
            mirrored_controller_data_dict[index] = mirrored_controller_data

        mirrored_pose = PoseInterpolatorPoseData(
            name=source_pose.name,
            rotations=source_pose.rotations,
            translations=source_pose.translations,
            independent=source_pose.independent,
            rotation_falloff=source_pose.rotation_falloff,
            translation_falloff=source_pose.translation_falloff,
            pose_type=source_pose.pose_type,
            gaussian_falloff=source_pose.gaussian_falloff,
            is_enabled=source_pose.is_enabled,
            controller_data=mirrored_controller_data_dict,
        )
        mirrored_poses.append(mirrored_pose)

    mirrored_outputs = []
    for output_list in source_data.outputs:
        mirrored_output_list = []
        for output in output_list:
            mirrored_output = flip_side(output) if map_mirror is None else map_mirror(output)
            if not cmds.objExists(mirrored_output):
                raise NotImplementedError(
                    f"The mirrored output attribute {mirrored_output} doesn't exist! "
                    "This function doesn't yet support creating mirroring shapes or other outputs."
                )
            mirrored_output_list.append(mirrored_output)
        mirrored_outputs.append(mirrored_output_list)

    mirrored_data = PoseInterpolatorData(
        name=mirrored_name,
        allow_negative_weights=source_data.allow_negative_weights,
        enable_rotation=source_data.enable_rotation,
        enable_translation=source_data.enable_translation,
        interpolation=source_data.interpolation,
        output_smoothing=source_data.output_smoothing,
        regularization=source_data.regularization,
        drivers=mirrored_drivers,
        poses=mirrored_poses,
        outputs=mirrored_outputs,
    )

    apply_pose_interpolator_data(destination, mirrored_data)


def mirror_pose_interpolators(
    pose_interpolators: Iterable[PoseInterpolatorNode | str],
    map_mirror: Callable[[str], str] | None = None,
) -> list[PoseInterpolatorNode]:
    mirrored = []
    for pose_interpolator in pose_interpolators:
        pose_interpolator_node = resolve_pose_interpolator_node(pose_interpolator)
        pose_interpolator_string = str(pose_interpolator_node)
        mirrored_pose_interpolator = (
            flip_side(pose_interpolator_string)
            if map_mirror is None
            else map_mirror(pose_interpolator_string)
        )
        if pose_interpolator_string == mirrored_pose_interpolator:
            # TODO: Make this mirror the poses and names within the pose interpolator. For things like the neck.
            continue
        if cmds.objExists(mirrored_pose_interpolator):
            mirrored_pose_interpolator_node = resolve_pose_interpolator_node(
                mirrored_pose_interpolator
            )
        else:
            # TODO: Make this put the new poseInterpolator in the correct mirrored directory and parent
            manager = PoseInterpolatorManagerNode.from_existing("poseInterpolatorManager")
            _, mirrored_pose_interpolator_node = create_pose_interpolator(
                manager, mirrored_pose_interpolator
            )

        mirror_pose_interpolator(
            pose_interpolator_node, mirrored_pose_interpolator_node, map_mirror
        )
        mirrored.append(mirrored_pose_interpolator_node)
    return mirrored
