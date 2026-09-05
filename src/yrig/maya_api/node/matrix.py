from yrig.maya_api.attribute import (
    AimMatrixAxisAttribute,
    ArrayAttribute,
    BlendMatrixTargetAttribute,
    BooleanAttribute,
    EnumAttribute,
    IntegerAttribute,
    MatrixAttribute,
    QuatAttribute,
    ScalarAttribute,
    Vector3Attribute,
    Vector4Attribute,
    WtMatrixAttribute,
)
from yrig.maya_api.enum import Axis, RotateOrder

from .core import Node


class AimMatrixNode(Node):
    """Maya aimMatrix node with enhanced interface."""

    node_type = "aimMatrix"

    def __init__(self, name: str = "aimMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.primary = AimMatrixAxisAttribute(f"{self.name}.primary")
        self.secondary = AimMatrixAxisAttribute(f"{self.name}.secondary")
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")

        self.pre_space_matrix = MatrixAttribute(f"{self.name}.preSpaceMatrix")
        self.post_space_matrix = MatrixAttribute(f"{self.name}.postSpaceMatrix")


class AxisFromMatrixNode(Node):
    """Maya axisFromMatrix node with enhanced interface."""

    node_type = "axisFromMatrix"

    def __init__(self, name: str = "axisFromMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = MatrixAttribute(f"{self.name}.input")
        self.axis = EnumAttribute(f"{self.name}.axis", Axis)
        self.output = Vector3Attribute(f"{self.name}.output")


class BlendMatrixNode(Node):
    """Maya blendMatrix node with enhanced interface."""

    node_type = "blendMatrix"

    def __init__(self, name: str = "blendMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.post_space_matrix = MatrixAttribute(f"{self.name}.postSpaceMatrix")
        self.pre_space_matrix = MatrixAttribute(f"{self.name}.preSpaceMatrix")
        self.target = ArrayAttribute(f"{self.name}.target", BlendMatrixTargetAttribute)
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")


class ComposeMatrixNode(Node):
    """Maya composeMatrix node with enhanced interface."""

    node_type = "composeMatrix"

    def __init__(self, name: str = "composeMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:

        self.input_rotate_order = EnumAttribute(f"{self.name}.inputRotateOrder", RotateOrder)
        self.input_quat = QuatAttribute(f"{self.name}.inputQuat")
        self.input_rotate = Vector3Attribute(f"{self.name}.inputRotate")
        self.input_scale = Vector3Attribute(f"{self.name}.inputScale")
        self.input_shear = Vector3Attribute(f"{self.name}.inputShear")
        self.input_translate = Vector3Attribute(f"{self.name}.inputTranslate")
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")


class DecomposeMatrixNode(Node):
    """Maya decomposeMatrix node with enhanced interface."""

    node_type = "decomposeMatrix"

    def __init__(self, name: str = "decomposeMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.input_rotate_order = EnumAttribute(f"{self.name}.inputRotateOrder", RotateOrder)
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")
        self.output_rotate = Vector3Attribute(f"{self.name}.outputRotate")
        self.output_scale = Vector3Attribute(f"{self.name}.outputScale")
        self.output_shear = Vector3Attribute(f"{self.name}.outputShear")
        self.output_translate = Vector3Attribute(f"{self.name}.outputTranslate")


class FourByFourMatrixNode(Node):
    """Maya fourByFourMatrix node with enhanced interface."""

    node_type = "fourByFourMatrix"

    def __init__(self, name: str = "fourByFourMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.in_00 = ScalarAttribute(f"{self.name}.in00")
        self.in_01 = ScalarAttribute(f"{self.name}.in01")
        self.in_02 = ScalarAttribute(f"{self.name}.in02")
        self.in_03 = ScalarAttribute(f"{self.name}.in03")
        self.in_10 = ScalarAttribute(f"{self.name}.in10")
        self.in_11 = ScalarAttribute(f"{self.name}.in11")
        self.in_12 = ScalarAttribute(f"{self.name}.in12")
        self.in_13 = ScalarAttribute(f"{self.name}.in13")
        self.in_20 = ScalarAttribute(f"{self.name}.in20")
        self.in_21 = ScalarAttribute(f"{self.name}.in21")
        self.in_22 = ScalarAttribute(f"{self.name}.in22")
        self.in_23 = ScalarAttribute(f"{self.name}.in23")
        self.in_30 = ScalarAttribute(f"{self.name}.in30")
        self.in_31 = ScalarAttribute(f"{self.name}.in31")
        self.in_32 = ScalarAttribute(f"{self.name}.in32")
        self.in_33 = ScalarAttribute(f"{self.name}.in33")
        self.output = MatrixAttribute(f"{self.name}.output")


class InverseMatrixNode(Node):
    """Maya inverseMatrix node with enhanced interface."""

    node_type = "inverseMatrix"
    plugin = "matrixNodes"

    def __init__(self, name: str = "inverseMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")


class MultiplyPointByMatrixNode(Node):
    """Maya multiplyPointByMatrix node with enhanced interface."""

    node_type = "multiplyPointByMatrix"

    def __init__(self, name: str = "multiplyPointByMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_point = Vector3Attribute(f"{self.name}.input")
        self.input_matrix = MatrixAttribute(f"{self.name}.matrix")
        self.output = Vector3Attribute(f"{self.name}.output")


class MultiplyVectorByMatrixNode(Node):
    """Maya multiplyVectorByMatrix node with enhanced interface."""

    node_type = "multiplyVectorByMatrix"

    def __init__(self, name: str = "multiplyVectorByMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_vector = Vector3Attribute(f"{self.name}.input")
        self.input_matrix = MatrixAttribute(f"{self.name}.matrix")
        self.output = Vector3Attribute(f"{self.name}.output")


class MultMatrixNode(Node):
    """Maya multMatrix node with enhanced interface."""

    node_type = "multMatrix"

    def __init__(self, name: str = "multMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.matrix_in = ArrayAttribute(f"{self.name}.matrixIn", MatrixAttribute)
        self.matrix_sum = MatrixAttribute(f"{self.name}.matrixSum")


class PickMatrixNode(Node):
    """Maya pickMatrix node with enhanced interface."""

    node_type = "pickMatrix"

    def __init__(self, name: str = "pickMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.use_translate = BooleanAttribute(f"{self.name}.useTranslate")
        self.use_rotate = BooleanAttribute(f"{self.name}.useRotate")
        self.use_scale = BooleanAttribute(f"{self.name}.useScale")
        self.use_shear = BooleanAttribute(f"{self.name}.useShear")
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")


class RowFromMatrixNode(Node):
    """Maya rowFromMatrix node with enhanced interface."""

    node_type = "rowFromMatrix"

    def __init__(self, name: str = "rowFromMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input = IntegerAttribute(f"{self.name}.input")
        self.matrix = MatrixAttribute(f"{self.name}.matrix")
        self.output = Vector4Attribute(f"{self.name}.output")


class WtAddMatrixNode(Node):
    """Maya wtAddMatrix node with enhanced interface."""

    node_type = "wtAddMatrix"

    def __init__(self, name: str = "wtAddMatrix") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.weight_matrix = ArrayAttribute(f"{self.name}.wtMatrix", WtMatrixAttribute)
        self.matrix_sum = MatrixAttribute(f"{self.name}.matrixSum")
