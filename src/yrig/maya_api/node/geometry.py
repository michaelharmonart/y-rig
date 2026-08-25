from yrig.maya_api.attribute import (
    ArrayAttribute,
    BooleanAttribute,
    ClosestPointOnSurfaceResultAttribute,
    EnumAttribute,
    GeometryAttribute,
    MatrixAttribute,
    NurbsCurveAttribute,
    NurbsSurfaceAttribute,
    ScalarAttribute,
    StringAttribute,
    UvPinCoordinateAttribute,
    Vector3Attribute,
)
from yrig.maya_api.enum import (
    Axis,
    MotionPathWorldUpType,
    RotateOrder,
    UnsignedAxis,
    UvPinNormalOverride,
    UvPinRelativeSpaceMode,
)

from .core import Node


class ClosestPointOnSurfaceNode(Node):
    """Maya closestPointOnSurface node with enhanced interface."""

    node_type = "closestPointOnSurface"

    def __init__(self, name: str = "closestPointOnSurface") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_surface = NurbsSurfaceAttribute(f"{self.name}.inputSurface")
        self.in_position = Vector3Attribute(f"{self.name}.inPosition")
        self.result = ClosestPointOnSurfaceResultAttribute(f"{self.name}.result")


class CurveInfoNode(Node):
    """Maya curveInfo node with enhanced interface."""

    node_type = "curveInfo"

    def __init__(self, name: str = "curveInfo") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_curve = NurbsCurveAttribute(f"{self.name}.inputCurve")
        self.arc_length = ScalarAttribute(f"{self.name}.arcLength")
        self.control_points = ArrayAttribute(f"{self.name}.controlPoints", ScalarAttribute)
        self.knots = ArrayAttribute(f"{self.name}.knots", ScalarAttribute)
        self.weights = ArrayAttribute(f"{self.name}.weights", ScalarAttribute)


class MotionPathNode(Node):
    """Maya motionPath node with enhanced interface."""

    node_type = "motionPath"

    def __init__(self, name: str = "motionPath") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:

        self.geometry_path = NurbsCurveAttribute(f"{self.name}.geometryPath")
        self.rotate_order = EnumAttribute(f"{self.name}.rotateOrder", RotateOrder)

        self.u_value = ScalarAttribute(f"{self.name}.uValue")
        self.fraction_mode = BooleanAttribute(f"{self.name}.fractionMode")

        self.follow = BooleanAttribute(f"{self.name}.follow")
        self.world_up_type = EnumAttribute(f"{self.name}.worldUpType", MotionPathWorldUpType)
        self.world_up_vector = Vector3Attribute(f"{self.name}.worldUpVector")
        self.world_up_matrix = MatrixAttribute(f"{self.name}.worldUpMatrix")
        self.inverse_up = BooleanAttribute(f"{self.name}.inverseUp")
        self.inverse_front = BooleanAttribute(f"{self.name}.inverseFront")
        self.front_axis = EnumAttribute(f"{self.name}.frontAxis", UnsignedAxis)
        self.up_axis = EnumAttribute(f"{self.name}.upAxis", UnsignedAxis)

        self.front_twist = ScalarAttribute(f"{self.name}.frontTwist")
        self.up_twist = ScalarAttribute(f"{self.name}.upTwist")
        self.side_twist = ScalarAttribute(f"{self.name}.sideTwist")

        self.bank = BooleanAttribute(f"{self.name}.bank")
        self.bank_limit = ScalarAttribute(f"{self.name}.bankLimit")
        self.bank_scale = ScalarAttribute(f"{self.name}.bankScale")

        self.all_coordinates = Vector3Attribute(f"{self.name}.allCoordinates")
        self.orient_matrix = MatrixAttribute(f"{self.name}.orientMatrix")
        self.rotate = Vector3Attribute(f"{self.name}.rotate")


class UvPinNode(Node):
    """Maya uvPin node with enhanced interface."""

    node_type = "uvPin"

    def __init__(self, name: str = "uvPin") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.original_geometry = GeometryAttribute(f"{self.name}.originalGeometry")
        self.deformed_geometry = GeometryAttribute(f"{self.name}.deformedGeometry")

        self.normal_axis = EnumAttribute(f"{self.name}.normalAxis", Axis)
        self.tangent_axis = EnumAttribute(f"{self.name}.tangentAxis", Axis)
        self.uv_set_name = StringAttribute(f"{self.name}.uvSetName")
        self.normalized_isoparms = BooleanAttribute(f"{self.name}.normalizedIsoParms")
        self.normal_override = EnumAttribute(f"{self.name}.normalOverride", UvPinNormalOverride)
        self.relative_space_mode = EnumAttribute(
            f"{self.name}.relativeSpaceMode", UvPinRelativeSpaceMode
        )
        self.relative_space_matrix = MatrixAttribute(f"{self.name}.relativeSpaceMatrix")
        self.coordinate = ArrayAttribute(f"{self.name}.coordinate", UvPinCoordinateAttribute)
        self.output_matrix = ArrayAttribute(f"{self.name}.outputMatrix", MatrixAttribute)
        self.output_translate = ArrayAttribute(f"{self.name}.outputTranslate", Vector3Attribute)
