from .core import ColorAttribute, ScalarAttribute


class SkinClusterInfluenceColor(ColorAttribute):
    """A Maya attribute of the same compound type as the blendShape Inbetween Info."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)
        self.r = ScalarAttribute(f"{attr_path}.influenceColorR")
        self.g = ScalarAttribute(f"{attr_path}.influenceColorG")
        self.b = ScalarAttribute(f"{attr_path}.influenceColorB")
