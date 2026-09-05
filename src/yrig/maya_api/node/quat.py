from yrig.maya_api.attribute import EnumAttribute, QuatAttribute, ScalarAttribute, Vector3Attribute
from yrig.maya_api.enum import RotateOrder

from .core import Node


class EulerToQuatNode(Node):
    """Maya eulerToQuat node with enhanced interface."""

    node_type = "eulerToQuat"
    plugin = "quatNodes"

    def __init__(self, name: str = "eulerToQuat") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")
        self.input_rotate_order = EnumAttribute(f"{self.name}.inputRotateOrder", RotateOrder)
        self.input_rotate = Vector3Attribute(f"{self.name}.inputRotate")


class QuatInvertNode(Node):
    """Maya quatInvert node with enhanced interface."""

    node_type = "quatInvert"
    plugin = "quatNodes"

    def __init__(self, name: str = "quatInvert") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_quat = QuatAttribute(f"{self.name}.inputQuat")
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")


class QuatNormalizeNode(Node):
    """Maya quatNormalize node with enhanced interface."""

    node_type = "quatNormalize"
    plugin = "quatNodes"

    def __init__(self, name: str = "quatNormalize") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_quat = QuatAttribute(f"{self.name}.inputQuat")
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")


class QuatProdNode(Node):
    """Maya quatProd node with enhanced interface."""

    node_type = "quatProd"
    plugin = "quatNodes"

    def __init__(self, name: str = "quatProd") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input1_quat = QuatAttribute(f"{self.name}.input1Quat")
        self.input2_quat = QuatAttribute(f"{self.name}.input2Quat")
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")


class QuatSlerpNode(Node):
    """Maya quatSlerp node with enhanced interface."""

    node_type = "quatSlerp"
    plugin = "quatNodes"

    def __init__(self, name: str = "quatSlerp") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input1_quat = QuatAttribute(f"{self.name}.input1Quat")
        self.input2_quat = QuatAttribute(f"{self.name}.input2Quat")
        self.input_t = ScalarAttribute(f"{self.name}.inputT")
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")


class QuatToEulerNode(Node):
    """Maya quatToEuler node with enhanced interface."""

    node_type = "quatToEuler"

    def __init__(self, name: str = "quatToEuler") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.input_quat = QuatAttribute(f"{self.name}.inputQuat")
        self.input_rotate_order = EnumAttribute(f"{self.name}.inputRotateOrder", RotateOrder)
        self.output_rotate = Vector3Attribute(f"{self.name}.outputRotate")
