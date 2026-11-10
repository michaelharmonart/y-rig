from yrig.maya_api.attribute import (
    AnimCurveKeyTimeValueAttribute,
    ArrayAttribute,
    BooleanAttribute,
    ColorAttribute,
    DoubleAngleAttribute,
    DoubleArrayAttribute,
    DoubleAttribute,
    DoubleLinearAttribute,
    EnumAttribute,
    FunctionAttribute,
    ScalarAttribute,
    StringAttribute,
    TimeAttribute,
)
from yrig.maya_api.enum import (
    AnimCurveStipplePattern,
    KeyframeInfinityType,
    KeyframeRotationInterpolation,
    KeyframeTangentType,
)

from .core import Node


class AnimCurveNode(Node):
    """Maya blendShape node with enhanced interface."""

    node_type = "animCurve"

    def __init__(self, name: str = "animCurve") -> None:
        super().__init__(name)

        self.input = ScalarAttribute(f"{self.name}.input")
        self.output = ScalarAttribute(f"{self.name}.output")

        self.apply = FunctionAttribute(f"{self.name}.baseOrigin")
        self.bin_membership = StringAttribute(f"{self.name}.binMembership")
        self.curve_color = ColorAttribute(f"{self.name}.curveColor")
        self.in_stipple_range = DoubleArrayAttribute(f"{self.name}.inStippleRange")
        self.key_breakdown = ArrayAttribute(f"{self.name}.keyBreakdown", BooleanAttribute)
        self.key_tan_in_type = ArrayAttribute(
            f"{self.name}.keyTanInType", lambda path: EnumAttribute(path, KeyframeTangentType)
        )
        self.key_tan_in_x = ArrayAttribute(f"{self.name}.keyTanInX", DoubleAttribute)
        self.key_tan_in_y = ArrayAttribute(f"{self.name}.keyTanInY", DoubleAttribute)
        self.key_tan_locked = ArrayAttribute(f"{self.name}.keyTanLocked", BooleanAttribute)
        self.key_tan_out_type = ArrayAttribute(
            f"{self.name}.keyTanOutType", lambda path: EnumAttribute(path, KeyframeTangentType)
        )
        self.key_tan_out_x = ArrayAttribute(f"{self.name}.keyTanOutX", DoubleAttribute)
        self.key_tan_out_y = ArrayAttribute(f"{self.name}.keyTanOutY", DoubleAttribute)
        self.key_tick_draw_special = BooleanAttribute(f"{self.name}.keyTickDrawSpecial")
        self.key_weight_locked = ArrayAttribute(f"{self.name}.keyWeightLocked", BooleanAttribute)
        self.out_stipple_range = DoubleArrayAttribute(f"{self.name}.outStippleRange")
        self.out_stipple_threshold = DoubleAttribute(f"{self.name}.outStippleThreshold")
        self.post_infinity = EnumAttribute(f"{self.name}.postInfinity", KeyframeInfinityType)
        self.pre_infinity = EnumAttribute(f"{self.name}.preInfinity", KeyframeInfinityType)
        self.rotation_interpolation = EnumAttribute(
            f"{self.name}.rotationInterpolation", KeyframeRotationInterpolation
        )
        self.stipple_pattern = EnumAttribute(f"{self.name}.stipplePattern", AnimCurveStipplePattern)
        self.stipple_reverse = BooleanAttribute(f"{self.name}.stippleReverse")
        self.tangent_type = EnumAttribute(f"{self.name}.tangentType", KeyframeTangentType)
        self.use_curve_color = BooleanAttribute(f"{self.name}.useCurveColor")
        self.weighted_tangents = BooleanAttribute(f"{self.name}.weightedTangents")


class AnimCurveTA(AnimCurveNode):
    node_type = "animCurveTA"

    def __init__(self, name: str = "animCurveTA") -> None:
        super().__init__(name)

        self.input = TimeAttribute(f"{self.name}.input")
        self.output = DoubleAngleAttribute(f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(path, TimeAttribute, DoubleAngleAttribute),
        )
        self.key_value_switch = TimeAttribute(f"{self.name}.keyValueSwitch")
        self.qauternion_w = DoubleAttribute(f"{self.name}.quaternionW")
        self.raw_value = DoubleAttribute(f"{self.name}.rawValue")


class AnimCurveTL(AnimCurveNode):
    node_type = "animCurveTL"

    def __init__(self, name: str = "animCurveTL") -> None:
        super().__init__(name)

        self.input = TimeAttribute(f"{self.name}.input")
        self.output = DoubleLinearAttribute(f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(path, TimeAttribute, DoubleLinearAttribute),
        )
        self.key_value_switch = TimeAttribute(f"{self.name}.keyValueSwitch")


class AnimCurveTT(AnimCurveNode):
    node_type = "animCurveTT"

    def __init__(self, name: str = "animCurveTT") -> None:
        super().__init__(name)

        self.input = TimeAttribute(f"{self.name}.input")
        self.output = TimeAttribute(f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(path, TimeAttribute, TimeAttribute),
        )
        self.key_value_switch = TimeAttribute(f"{self.name}.keyValueSwitch")


class AnimCurveTU(AnimCurveNode):
    node_type = "animCurveTU"

    def __init__(self, name: str = "animCurveTU") -> None:
        super().__init__(name)

        self.input = TimeAttribute(f"{self.name}.input")
        self.output = DoubleAttribute(attr_path=f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(path, TimeAttribute, DoubleAttribute),
        )
        self.key_value_switch = TimeAttribute(f"{self.name}.keyValueSwitch")


class AnimCurveUA(AnimCurveNode):
    node_type = "animCurveUA"

    def __init__(self, name: str = "animCurveUA") -> None:
        super().__init__(name)

        self.input = DoubleAttribute(f"{self.name}.input")
        self.output = DoubleAngleAttribute(attr_path=f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(
                path, DoubleAttribute, DoubleAngleAttribute
            ),
        )


class AnimCurveUL(AnimCurveNode):
    node_type = "animCurveUL"

    def __init__(self, name: str = "animCurveUL") -> None:
        super().__init__(name)

        self.input = DoubleAttribute(f"{self.name}.input")
        self.output = DoubleLinearAttribute(attr_path=f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(
                path, DoubleAttribute, DoubleLinearAttribute
            ),
        )


class AnimCurveUT(AnimCurveNode):
    node_type = "animCurveUT"

    def __init__(self, name: str = "animCurveUT") -> None:
        super().__init__(name)

        self.input = DoubleAttribute(f"{self.name}.input")
        self.output = TimeAttribute(attr_path=f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(path, DoubleAttribute, TimeAttribute),
        )


class AnimCurveUU(AnimCurveNode):
    node_type = "animCurveUU"

    def __init__(self, name: str = "animCurveUU") -> None:
        super().__init__(name)

        self.input = DoubleAttribute(f"{self.name}.input")
        self.output = DoubleAttribute(f"{self.name}.output")
        self.key_time_value = ArrayAttribute(
            f"{self.name}.keyTimeValue",
            lambda path: AnimCurveKeyTimeValueAttribute(path, DoubleAttribute, DoubleAttribute),
        )
