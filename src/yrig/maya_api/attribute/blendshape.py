from yrig.maya_api.enum import (
    BlendShapeInbetweenTargetType,
    BlendShapeInterpolation,
    BlendShapePostDeformationOrder,
)

from .core import (
    ArrayAttribute,
    Attribute,
    BooleanAttribute,
    ComponentListAttribute,
    EnumAttribute,
    GeometryAttribute,
    Int32ArrayAttribute,
    IntegerAttribute,
    MatrixArrayAttribute,
    MatrixAttribute,
    PointArrayAttribute,
    ScalarAttribute,
    StringAttribute,
    Vector2Attribute,
    Vector3Attribute,
)


class BlendShapeInbetweenInfoAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape Inbetween Info."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)
        self.inbetween_target_type = EnumAttribute(
            f"{attr_path}.inbetweenTargetType", BlendShapeInbetweenTargetType
        )
        self.inbetween_target_name = StringAttribute(f"{attr_path}.inbetweenTargetName")
        self.interpolation = EnumAttribute(f"{attr_path}.interpolation", BlendShapeInterpolation)
        self.interpolation_curve = ArrayAttribute(
            f"{attr_path}.interpolationCurve", Vector2Attribute
        )
        self.inbetween_visibility = BooleanAttribute(f"{attr_path}.inbetweenVisibility")


class BlendShapeInbetweenInfoGroupAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape Inbetween Info Group."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.inbetween_info = BlendShapeInbetweenInfoAttribute(f"{attr_path}.inbetweenInfoGroup")


class BlendShapeInputAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape Input."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.input_geometry = GeometryAttribute(f"{attr_path}.inputGeometry")
        self.group_id = IntegerAttribute(f"{attr_path}.groupId")
        self.component_tag_expression = StringAttribute(f"{attr_path}.componentTagExpression")


class BlendShapeInputTargetAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape inputTarget."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.input_target_group = ArrayAttribute(
            f"{attr_path}.inputTargetGroup", BlendShapeInputTargetGroupAttribute
        )
        self.base_weights = ArrayAttribute(f"{attr_path}.baseWeights", ScalarAttribute)
        self.normalization_group = ArrayAttribute(
            f"{attr_path}.normalizationGroup", BlendShapeNormalizationGroupAttribute
        )
        self.paint_target_weights = ArrayAttribute(
            f"{attr_path}.paintTargetWeights", ScalarAttribute
        )
        self.paint_target_index = IntegerAttribute(f"{attr_path}.paintTargetIndex")
        self.sculpt_target_index = IntegerAttribute(f"{attr_path}.sculptTargetIndex")
        self.sculpt_inbetween_weight = ScalarAttribute(f"{attr_path}.sculptInbetweenWeight")
        self.sculpt_target_tweaks = ArrayAttribute(
            f"{attr_path}.sculptTargetTweaks", BlendShapeSculptTargetTweaksAttribute
        )
        self.deform_matrix = MatrixArrayAttribute(f"{attr_path}.deformMatrix")
        self.deform_matrix_modified = BooleanAttribute(f"{attr_path}.deformMatrixModified")


class BlendShapeInputTargetGroupAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape inputTargetGroup."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.input_target_item = ArrayAttribute(
            f"{attr_path}.inputTargetItem", BlendShapeInputTargetItemAttribute
        )
        self.target_weights = ArrayAttribute(f"{attr_path}.targetWeights", ScalarAttribute)
        self.normalization_id = IntegerAttribute(f"{attr_path}.normalizationId")
        self.post_deformers_mode = EnumAttribute(
            f"{attr_path}.postDeformersMode", BlendShapePostDeformationOrder
        )
        self.target_bind_matrix = MatrixAttribute(f"{attr_path}.targetBindMatrix")
        self.target_matrix = MatrixAttribute(f"{attr_path}.targetMatrix")


class BlendShapeInputTargetItemAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape inputTargetItem."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.input_geom_target = GeometryAttribute(f"{attr_path}.inputGeomTarget")
        self.input_relative_points_target = PointArrayAttribute(
            f"{attr_path}.inputRelativePointsTarget"
        )
        self.input_relative_components_target = ComponentListAttribute(
            f"{attr_path}.inputRelativeComponentsTarget"
        )
        self.input_points_target = PointArrayAttribute(f"{attr_path}.inputPointsTarget")
        self.input_components_target = ComponentListAttribute(f"{attr_path}.inputComponentsTarget")


class BlendShapeNormalizationGroupAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape Normalization Group."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.normalization_use_weights = BooleanAttribute(f"{attr_path}.normalizationUseWeights")
        self.normalization_weights = ArrayAttribute(
            f"{attr_path}.normalizationWeights", ScalarAttribute
        )


class BlendShapeSculptTargetTweaksAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape sculpt target tweaks."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.vertex = ArrayAttribute(f"{attr_path}.vertex", Vector3Attribute)
        self.control_points = ArrayAttribute(f"{attr_path}.controlPointe", Vector3Attribute)


class BlendShapeTargetDirectoryAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape TargetDirectory."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.child_indices = Int32ArrayAttribute(f"{attr_path}.childIndices")
        self.parent_index = IntegerAttribute(f"{attr_path}.parentIndex")
        self.directory_name = StringAttribute(f"{attr_path}.directoryName")
        self.directory_visibility = BooleanAttribute(f"{attr_path}.directoryVisibility")
        self.directory_parent_visibility = BooleanAttribute(
            f"{attr_path}.directoryParentVisibility"
        )
        self.directory_weight = ScalarAttribute(f"{attr_path}.directoryWeight")


class BlendShapeWeightListAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape WeightList."""

    def __init__(self, attr_path: str) -> None:
        super().__init__(attr_path)

        self.weights = ArrayAttribute(f"{attr_path}.weights", ScalarAttribute)


class BlendShapeWeightFunctionDataAttribute(Attribute):
    """A Maya attribute of the same compound type as the blendShape Weight Function Data."""
