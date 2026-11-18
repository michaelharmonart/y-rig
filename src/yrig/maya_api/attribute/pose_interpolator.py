from yrig.maya_api.enum import (
    PoseInterpolatorPoseControllerDataItemType,
    PoseInterpolatorPoseType,
    RotateOrder,
    UnsignedAxis,
)

from .core import (
    ArrayAttribute,
    Attribute,
    BooleanAttribute,
    DoubleArrayAttribute,
    EnumAttribute,
    EulerRotationAttribute,
    FloatAttribute,
    GenericAttribute,
    Int32ArrayAttribute,
    LongAttribute,
    MatrixAttribute,
    StringAttribute,
)


class PoseInterpolatorDirectoryAttribute(Attribute):
    """A Maya attribute of the same compound type as the poseInterpolator directory."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.child_indices = Int32ArrayAttribute(f"{attr_path}.childIndices")
        self.parent_index = LongAttribute(f"{attr_path}.parentIndex")
        self.directory_name = StringAttribute(f"{attr_path}.directoryName")


class PoseInterpolatorDriverAttribute(Attribute):
    """A Maya attribute of the same compound type as the poseInterpolator driver."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.driver_matrix = MatrixAttribute(f"{attr_path}.driverMatrix")
        self.driver_orient = EulerRotationAttribute(f"{attr_path}.driverOrient")
        self.driver_rotate_axis = EulerRotationAttribute(f"{attr_path}.driverRotateAxis")
        self.driver_twist_axis = EnumAttribute(f"{attr_path}.driverTwistAxis", UnsignedAxis)
        self.driver_rotate_order = EnumAttribute(f"{attr_path}.driverRotateOrder", RotateOrder)
        self.driver_euler_twist = BooleanAttribute(f"{attr_path}.driverEulerTwist")
        self.driver_controller = ArrayAttribute(f"{attr_path}.driverController", GenericAttribute)


class PoseInterpolatorPoseControllerDataItemAttribute(Attribute):
    """A Maya attribute of the same compound type as the poseInterpolator poseControllerDataItem."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.pose_controller_data_item_name = StringAttribute(
            f"{attr_path}.poseControllerDataItemName"
        )
        self.pose_controller_data_item_type = EnumAttribute(
            f"{attr_path}.poseControllerDataItemType",
            PoseInterpolatorPoseControllerDataItemType,
        )
        self.pose_controller_data_item_value = GenericAttribute(
            f"{attr_path}.poseControllerDataItemValue"
        )


class PoseInterpolatorPoseControllerDataAttribute(Attribute):
    """A Maya attribute of the same compound type as the poseInterpolator poseControllerData."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.pose_controller_data_item = ArrayAttribute(
            f"{attr_path}.poseControllerDataItem", PoseInterpolatorPoseControllerDataItemAttribute
        )


class PoseInterpolatorPoseAttribute(Attribute):
    """A Maya attribute of the same compound type as the poseInterpolator pose."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.pose_rotation = ArrayAttribute(f"{attr_path}.poseRotation", DoubleArrayAttribute)
        self.pose_translation = ArrayAttribute(f"{attr_path}.poseTranslation", DoubleArrayAttribute)
        self.pose_name = StringAttribute(f"{attr_path}.poseName")
        self.is_independent = BooleanAttribute(f"{attr_path}.isIndependent")
        self.pose_rotation_falloff = FloatAttribute(f"{attr_path}.poseRotationFalloff")
        self.pose_translation_falloff = FloatAttribute(f"{attr_path}.poseTranslationFalloff")
        self.pose_type = EnumAttribute(f"{attr_path}.poseType", PoseInterpolatorPoseType)
        self.pose_falloff = FloatAttribute(f"{attr_path}.poseFalloff")
        self.is_enabled = BooleanAttribute(f"{attr_path}.isEnabled")
        self.pose_controller_data = ArrayAttribute(
            f"{attr_path}.poseControllerData", PoseInterpolatorPoseControllerDataAttribute
        )
