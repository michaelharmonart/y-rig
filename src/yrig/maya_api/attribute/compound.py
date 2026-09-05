from maya import cmds

from yrig.maya_api.enum import AimMatrixAxisMode

from .core import (
    Attribute,
    BooleanAttribute,
    EnumAttribute,
    Int32ArrayAttribute,
    IntegerAttribute,
    MatrixAttribute,
    ScalarAttribute,
    StringAttribute,
    Vector3Attribute,
)


class AimMatrixAxisAttribute(Attribute):
    """A Maya attribute of the same compound type as the aimMatrix axes."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.input_axis = Vector3Attribute(f"{attr_path}InputAxis")
        self.mode = EnumAttribute(f"{attr_path}Mode", AimMatrixAxisMode)
        self.target_vector = Vector3Attribute(f"{attr_path}TargetVector")
        self.target_matrix = MatrixAttribute(f"{attr_path}TargetMatrix")


class BlendMatrixTargetAttribute(Attribute):
    """A Maya attribute of the same compound type as the targets in a blendMatrix node."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.target_matrix = MatrixAttribute(f"{attr_path}.targetMatrix")
        self.use_matrix = BooleanAttribute(f"{attr_path}.useMatrix")
        self.weight = ScalarAttribute(f"{attr_path}.weight")
        self.scale_weight = ScalarAttribute(f"{attr_path}.scaleWeight")
        self.translate_weight = ScalarAttribute(f"{attr_path}.translateWeight")
        self.rotate_weight = ScalarAttribute(f"{attr_path}.rotateWeight")
        self.shear_weight = ScalarAttribute(f"{attr_path}.shearWeight")


class ClosestPointOnSurfaceResultAttribute(Attribute):
    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.position = Vector3Attribute(f"{attr_path}.position")
        self.parameter_u = ScalarAttribute(f"{attr_path}.parameterU")
        self.parameter_v = ScalarAttribute(f"{attr_path}.parameterV")


class PoseInterpolatorDirectoryAttribute(Attribute):
    """A Maya attribute of the same compound type as the poseInterpolator directory."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.child_indices = Int32ArrayAttribute(f"{attr_path}.childIndices")
        self.parent_index = IntegerAttribute(f"{attr_path}.parentIndex")
        self.directory_name = StringAttribute(f"{attr_path}TargetVector")


class UvPinCoordinateAttribute(Attribute[tuple[float, float]]):
    """A Maya attribute of the type UV"""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.u = ScalarAttribute(f"{attr_path}.coordinateU")
        self.v = ScalarAttribute(f"{attr_path}.coordinateV")

    def get(self) -> tuple[float, float]:
        """Get the value of this attribute."""
        return_list = cmds.getAttr(self.attr_path)
        tuple = return_list[0]
        return tuple

    def set(self, value: tuple[float, float]) -> None:
        """Set the value of this attribute."""
        cmds.setAttr(self.attr_path, *value)  # type: ignore


class WtMatrixAttribute(Attribute):
    """A Maya attribute of the same compound type as the wtMatrix elements in a wtAddMatrix node."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.matrix_in = MatrixAttribute(f"{attr_path}.matrixIn")
        self.weight_in = ScalarAttribute(f"{attr_path}.weightIn")
