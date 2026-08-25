from yrig.maya_api.attribute import (
    ArrayAttribute,
    ColorAttribute,
    EnumAttribute,
    MatrixAttribute,
    ScalarAttribute,
    Vector2Attribute,
    Vector3Attribute,
)
from yrig.maya_api.enum import (
    ConditionOperation,
    MultiplyDivideOperation,
    PlusMinusAverageOperation,
)

from .core import Node


class AbsoluteNode(Node):
    """Maya absolute node with enhanced interface."""

    node_type = "absolute"

    def __init__(self, name: str = "absolute") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = ScalarAttribute(f"{self.name}.input")
        self.output = ScalarAttribute(f"{self.name}.output")


class AddDLNode(Node):
    """Maya addDL node with enhanced interface."""

    node_type = "addDL"

    def __init__(self, name: str = "addDL") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_1 = ScalarAttribute(f"{self.name}.input1")
        self.input_2 = ScalarAttribute(f"{self.name}.input2")
        self.output = ScalarAttribute(f"{self.name}.output")


class BlendColorsNode(Node):
    """Maya blendColors node with enhanced interface."""

    node_type = "blendColors"

    def __init__(self, name: str = "blendColors") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.color1: ColorAttribute = ColorAttribute(f"{self.name}.color1")
        self.color2: ColorAttribute = ColorAttribute(f"{self.name}.color2")
        self.output: ColorAttribute = ColorAttribute(f"{self.name}.output")
        self.blender = ScalarAttribute(f"{self.name}.blender")


class ClampRangeNode(Node):
    """Maya clampRange node with enhanced interface."""

    node_type = "clampRange"

    def __init__(self, name: str = "clampRange") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = ScalarAttribute(f"{self.name}.input")
        self.minimum = ScalarAttribute(f"{self.name}.minimum")
        self.maximum = ScalarAttribute(f"{self.name}.maximum")
        self.output = ScalarAttribute(f"{self.name}.output")


class ConditionNode(Node):
    """Maya condition node with enhanced interface."""

    node_type = "condition"

    def __init__(self, name: str = "condition") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.first_term = ScalarAttribute(f"{self.name}.firstTerm")
        self.second_term = ScalarAttribute(f"{self.name}.secondTerm")
        self.color_if_true = ColorAttribute(f"{self.name}.colorIfTrue")
        self.color_if_false = ColorAttribute(f"{self.name}.colorIfFalse")
        self.operation = EnumAttribute(f"{self.name}.operation", ConditionOperation)
        self.out_color: ColorAttribute = ColorAttribute(f"{self.name}.outColor")


class CosNode(Node):
    """Maya cos node with enhanced interface."""

    node_type = "cos"

    def __init__(self, name: str = "cos") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input: ScalarAttribute = ScalarAttribute(f"{self.name}.input")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")


class CrossProductNode(Node):
    """Maya crossProduct node with enhanced interface."""

    node_type = "crossProduct"

    def __init__(self, name: str = "crossProduct") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input1 = Vector3Attribute(f"{self.name}.input1")
        self.input2 = Vector3Attribute(f"{self.name}.input2")
        self.output = Vector3Attribute(f"{self.name}.output")


class DistanceBetweenNode(Node):
    """Maya distanceBetween node with enhanced interface."""

    node_type = "distanceBetween"

    def __init__(self, name: str = "distanceBetween") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.point1 = Vector3Attribute(f"{self.name}.point1")
        self.point2 = Vector3Attribute(f"{self.name}.point2")
        self.input_matrix1 = MatrixAttribute(f"{self.name}.inMatrix1")
        self.input_matrix2 = MatrixAttribute(f"{self.name}.inMatrix2")
        self.distance = ScalarAttribute(f"{self.name}.distance")


class DivideNode(Node):
    """Maya divide node with enhanced interface."""

    node_type = "divide"

    def __init__(self, name: str = "divide") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input1 = ScalarAttribute(f"{self.name}.input1")
        self.input2 = ScalarAttribute(f"{self.name}.input2")
        self.output = ScalarAttribute(f"{self.name}.output")


class LengthNode(Node):
    """Maya length node with enhanced interface."""

    node_type = "length"

    def __init__(self, name: str = "length") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = Vector3Attribute(f"{self.name}.input")
        self.output = ScalarAttribute(f"{self.name}.output")


class LerpNode(Node):
    """Maya lerp node with enhanced interface."""

    node_type = "lerp"

    def __init__(self, name: str = "lerp") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input1 = ScalarAttribute(f"{self.name}.input1")
        self.input2 = ScalarAttribute(f"{self.name}.input2")
        self.weight = ScalarAttribute(f"{self.name}.weight")
        self.output = ScalarAttribute(f"{self.name}.output")


class MultiplyNode(Node):
    """Maya multiply node with enhanced interface."""

    node_type = "multiply"

    def __init__(self, name: str = "multiply") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = ArrayAttribute(f"{self.name}.input", ScalarAttribute)
        self.output = ScalarAttribute(f"{self.name}.output")


class MultiplyDivideNode(Node):
    """Maya multiplyDivide node with enhanced interface."""

    node_type = "multiplyDivide"

    def __init__(self, name: str = "multiplyDivide") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input1 = Vector3Attribute(f"{self.name}.input1")
        self.input2 = Vector3Attribute(f"{self.name}.input2")
        self.operation = EnumAttribute(f"{self.name}.operation", MultiplyDivideOperation)
        self.output = Vector3Attribute(f"{self.name}.output")


class NormalizeNode(Node):
    """Maya normalize node with enhanced interface."""

    node_type = "normalize"

    def __init__(self, name: str = "normalize") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = Vector3Attribute(f"{self.name}.input")
        self.output = Vector3Attribute(f"{self.name}.output")


class PlusMinusAverageNode(Node):
    """Maya plusMinusAverage node with enhanced interface."""

    node_type = "plusMinusAverage"

    def __init__(self, name: str = "plusMinusAverage") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_3d = ArrayAttribute(f"{self.name}.input3D", Vector3Attribute)
        self.input_2d = ArrayAttribute(f"{self.name}.input2D", Vector2Attribute)
        self.input_1d = ArrayAttribute(f"{self.name}.input1D", ScalarAttribute)
        self.output_3d = Vector3Attribute(f"{self.name}.output3D")
        self.output_2d = Vector2Attribute(f"{self.name}.output2D")
        self.output_1d = ScalarAttribute(f"{self.name}.output1D")
        self.operation = EnumAttribute(f"{self.name}.operation", PlusMinusAverageOperation)


class RemapValueNode(Node):
    """Maya remapValue node with enhanced interface."""

    node_type = "remapValue"

    def __init__(self, name: str = "remapValue") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_value = ScalarAttribute(f"{self.name}.inputValue")
        self.output = ScalarAttribute(f"{self.name}.outValue")
        self.input_max = ScalarAttribute(f"{self.name}.inputMax")
        self.input_min = ScalarAttribute(f"{self.name}.inputMin")
        self.output_max = ScalarAttribute(f"{self.name}.outputMax")
        self.output_min = ScalarAttribute(f"{self.name}.outputMin")


class SinNode(Node):
    """Maya sin node with enhanced interface."""

    node_type = "sin"

    def __init__(self, name: str = "sin") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input: ScalarAttribute = ScalarAttribute(f"{self.name}.input")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")


class SubtractNode(Node):
    """Maya subtract node with enhanced interface."""

    node_type = "subtract"

    def __init__(self, name: str = "subtract") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input1: ScalarAttribute = ScalarAttribute(f"{self.name}.input1")
        self.input2: ScalarAttribute = ScalarAttribute(f"{self.name}.input2")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")


class SumNode(Node):
    """Maya sum node with enhanced interface."""

    node_type = "sum"

    def __init__(self, name: str = "sum") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = ArrayAttribute(f"{self.name}.input", ScalarAttribute)
        self.output = ScalarAttribute(f"{self.name}.output")
