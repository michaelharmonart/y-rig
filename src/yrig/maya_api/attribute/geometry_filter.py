from .core import (
    ArrayAttribute,
    Attribute,
    GeometryAttribute,
    IntegerAttribute,
    ScalarAttribute,
    StringAttribute,
)


class GeometryFilterInputAttribute(Attribute):
    """A Maya attribute of the same compound type as the geometryFilter Input."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.input_geometry = GeometryAttribute(f"{attr_path}.inputGeometry")
        self.group_id = IntegerAttribute(f"{attr_path}.groupId")
        self.component_tag_expression = StringAttribute(f"{attr_path}.componentTagExpression")


class GeometryFilterWeightFunctionDataAttribute(Attribute):
    """A Maya attribute of the same compound type as the geometryFilter Weight Function Data."""


class WeightGeometryFilterWeightListAttribute(Attribute):
    """A Maya attribute of the same compound type as the weightGeometryFilter WeightList."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.weights = ArrayAttribute(f"{attr_path}.weights", ScalarAttribute)
