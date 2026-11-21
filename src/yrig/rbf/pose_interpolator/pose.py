from maya import cmds
from maya.api.OpenMaya import MFnTransform, MQuaternion, MSpace, MVector

from yrig.maya_api.node import PoseInterpolatorNode
from yrig.maya_api.utils import get_dag_path

from .core import force_pose_interpolator_update, resolve_pose_interpolator_shape


def update_pose(pose_interpolator: PoseInterpolatorNode | str, pose_index: int) -> None:
    pose_interpolator_node = (
        pose_interpolator
        if isinstance(pose_interpolator, PoseInterpolatorNode)
        else PoseInterpolatorNode.from_existing(resolve_pose_interpolator_shape(pose_interpolator))
    )
    pose_attr = pose_interpolator_node.pose[pose_index]
    for driver_index in pose_interpolator_node.driver.get_indices():
        driver_attr = pose_interpolator_node.driver[driver_index]
        driver_obj_attr = driver_attr.driver_matrix.get_input()
        driver_obj = str(driver_obj_attr).split(".", 1)[0] if driver_obj_attr is not None else None
        if driver_obj is None or not cmds.objectType(driver_obj, isAType="transform"):
            raise NotImplementedError(
                "This function currently only supports updating poses for interpolators driven directly by a transform."
            )

        driver_transform: MFnTransform = MFnTransform(get_dag_path(driver_obj))

        driver_translation: MVector = driver_transform.translation(MSpace.kTransform)
        pose_translation_attr = pose_attr.pose_translation[driver_index]
        pose_translation_seq = [driver_translation.x, driver_translation.y, driver_translation.z]
        pose_translation_attr.set(pose_translation_seq)

        driver_rotation: MQuaternion = driver_transform.rotation(asQuaternion=True)
        pose_rotation_attr = pose_attr.pose_rotation[driver_index]
        pose_rotation_seq = [
            driver_rotation.x,
            driver_rotation.y,
            driver_rotation.z,
            driver_rotation.w,
        ]
        pose_rotation_attr.set(pose_rotation_seq)

        pose_controller_data_attr = pose_attr.pose_controller_data[driver_index]
        for controller_index in pose_controller_data_attr.pose_controller_data_item.get_indices():
            controller_item = pose_controller_data_attr.pose_controller_data_item[controller_index]
            controller_value = driver_attr.driver_controller[controller_index].get()
            controller_item.pose_controller_data_item_value.set(controller_value)
    force_pose_interpolator_update(pose_interpolator_node)
