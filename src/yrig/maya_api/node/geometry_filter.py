from yrig.maya_api.attribute import (
    ArrayAttribute,
    BooleanAttribute,
    GeometryAttribute,
    GeometryFilterInputAttribute,
    GeometryFilterWeightFunctionDataAttribute,
    Long3Attribute,
    ScalarAttribute,
    UInt64ArrayAttribute,
    WeightGeometryFilterWeightListAttribute,
)

from .core import Node


class GeometryFilter(Node):
    """Maya geometryFilter node with enhanced interface."""

    node_type = "geometryFilter"

    def __init__(self, name: str = "geometryFilter") -> None:
        super().__init__(name)
        self.block_gpu = BooleanAttribute(f"{self.name}.blockGPU")
        self.envelope = ScalarAttribute(f"{self.name}.envelope")
        self.input = ArrayAttribute(f"{self.name}.input", GeometryFilterInputAttribute)
        self.output_geometry = ArrayAttribute(
            f"{self.name}.outputGeometry", GeometryFilterInputAttribute
        )
        self.map_64_bit_indices = UInt64ArrayAttribute(f"{self.name}.map64BitIndices")
        self.original_geometry = ArrayAttribute(f"{self.name}.originalGeometry", GeometryAttribute)
        self.weight_function = ArrayAttribute(
            f"{self.name}.weightFunction", GeometryFilterWeightFunctionDataAttribute
        )
        self.function = Long3Attribute(f"{self.name}.function")


class WeightGeometryFilter(GeometryFilter):
    node_type = "weightGeometryFilter"

    def __init__(self, name: str = "weightGeometryFilter") -> None:
        super().__init__(name)
        self.weight_list = self.weight_list = ArrayAttribute(
            f"{self.name}.weightList", WeightGeometryFilterWeightListAttribute
        )
