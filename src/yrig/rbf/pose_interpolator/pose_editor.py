from dataclasses import dataclass

from maya import mel


@dataclass
class PoseInterpolatorPose:
    pose_interpolator: str
    pose_index: int


def get_selected_poses() -> list[PoseInterpolatorPose]:
    """Get the selected poses in Maya's Pose Editor."""
    poses = mel.eval("getPoseEditorTreeviewSelection(2)") or []

    pose_objects = []
    for pose in poses:
        interpolator, index = pose.rsplit(".", 1)
        index = int(index)
        pose_objects.append(PoseInterpolatorPose(interpolator, index))
    return pose_objects


def get_selected_pose_interpolators() -> list[str]:
    """Get the selected poseInterpolators in Maya's Pose Editor."""
    return mel.eval("getPoseEditorTreeviewSelection(1)") or []
