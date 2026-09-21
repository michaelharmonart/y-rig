from yrig.maya_api.attribute import (
    ArrayAttribute,
    BooleanAttribute,
    DoubleArrayAttribute,
    EnumAttribute,
    GeometryAttribute,
    IntegerAttribute,
    MatrixAttribute,
    MessageAttribute,
    ScalarAttribute,
    SkinClusterInfluenceColor,
    Vector3Attribute,
)
from yrig.maya_api.enum import (
    SkinClusterNormalizeWeights,
    SkinClusterRelativeSpaceMode,
    SkinClusterSkinningMethod,
    SkinClusterWeightDistribution,
)

from .core import Node


class SkinCluster(Node):
    """Maya skinCluster node with enhanced interface."""

    node_type = "skinCluster"

    def __init__(self, name: str = "skinCluster") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.base_dirty = MessageAttribute(f"{self.name}.baseDirty")
        self.basePoints = ArrayAttribute(f"{self.name}.basePoints", GeometryAttribute)
        self.bind_method = EnumAttribute(f"{self.name}.bindMethod", SkinClusterSkinningMethod)
        self.bind_pose = MessageAttribute(f"{self.name}.bindPose")
        self.bind_pre_matrix = ArrayAttribute(f"{self.name}.bindPreMatrix", MatrixAttribute)
        self.bind_volume = MessageAttribute(f"{self.name}.bindVolume")
        self.blend_weights = ArrayAttribute(f"{self.name}.blendWeights", ScalarAttribute)
        self.deform_user_normals = BooleanAttribute(f"{self.name}.deformUserNormals")
        self.dqs_scale = Vector3Attribute(f"{self.name}.dqsScale")
        self.driver_points = ArrayAttribute(f"{self.name}.driverPoints", GeometryAttribute)
        self.dropoff = ArrayAttribute(f"{self.name}.dropoff", ScalarAttribute)
        self.dropoff_rate = ScalarAttribute(f"{self.name}.dropoffRate")
        self.envelope = ScalarAttribute(f"{self.name}.envelope")
        self.geom_bind = MessageAttribute(f"{self.name}.geomBind")
        self.geom_matrix = MatrixAttribute(f"{self.name}.geomMatrix")
        self.heatmap_falloff = ScalarAttribute(f"{self.name}.heatmapFalloff")
        self.influence_color = ArrayAttribute(
            f"{self.name}.influenceColor", SkinClusterInfluenceColor
        )
        self.lock_weights = ArrayAttribute(f"{self.name}.lockWeights", BooleanAttribute)
        self.maintain_max_influences = BooleanAttribute(f"{self.name}.maintainMaxInfluences")
        self.matrix = ArrayAttribute(f"{self.name}.matrix", MatrixAttribute)
        self.max_influences = IntegerAttribute(f"{self.name}.maxInfluences")
        self.normalize_weights = EnumAttribute(
            f"{self.name}.normalizeWeights", SkinClusterNormalizeWeights
        )
        self.nurbs_samples = ArrayAttribute(f"{self.name}.nurbsSamples", IntegerAttribute)
        self.paint_arr_dirty = MessageAttribute(f"{self.name}.paintArrDirty")
        self.paint_trans = MessageAttribute(f"{self.name}.paintTrans")
        self.paint_weights = DoubleArrayAttribute(f"{self.name}.paintWeights")
        self.relative_space_matrix = MatrixAttribute(f"{self.name}.relativeSpaceMatrix")
        self.relative_space_mode = EnumAttribute(
            f"{self.name}.relativeSpaceMode", SkinClusterRelativeSpaceMode
        )
        self.skinning_method = EnumAttribute(
            f"{self.name}.skinningMethod", SkinClusterSkinningMethod
        )
        self.smoothness = ArrayAttribute(f"{self.name}.smoothness", ScalarAttribute)
        self.use_components = BooleanAttribute(f"{self.name}.useComponents")
        self.use_components_matrix = BooleanAttribute(f"{self.name}.useComponentsMatrix")
        self.weight_distribution = EnumAttribute(
            f"{self.name}.weightDistribution", SkinClusterWeightDistribution
        )
        self.wt_drty = MessageAttribute(f"{self.name}.wtDrty")
