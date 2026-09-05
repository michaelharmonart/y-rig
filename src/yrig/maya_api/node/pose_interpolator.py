from yrig.maya_api.attribute import ArrayAttribute
from yrig.maya_api.attribute.compound import PoseInterpolatorDirectoryAttribute
from yrig.maya_api.attribute.core import Int32ArrayAttribute

from .core import Node


class PoseInterpolatorManager(Node):
    """Maya poseInterpolatorManager node with enhanced interface."""

    node_type = "poseInterpolatorManager"
    plugin = "poseInterpolator"

    def __init__(self, name: str = "poseInterpolatorManager") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.pose_interpolator_directory = ArrayAttribute(
            f"{self.name}.poseInterpolatorDirectory", PoseInterpolatorDirectoryAttribute
        )
        self.pose_interpolator_parent = ArrayAttribute(
            f"{self.name}.poseInterpolatorParent", Int32ArrayAttribute
        )
