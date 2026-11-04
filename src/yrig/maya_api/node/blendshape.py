from yrig.maya_api.attribute import (
    ArrayAttribute,
    BlendShapeInbetweenInfoAttribute,
    BlendShapeInputTargetAttribute,
    BlendShapeTargetDirectoryAttribute,
    BooleanAttribute,
    DoubleArrayAttribute,
    EnumAttribute,
    IntegerAttribute,
    LongAttribute,
    ScalarAttribute,
    ShapeEditorManagerDirectoryAttribute,
    StringAttribute,
    Vector3Attribute,
)
from yrig.maya_api.enum import BlendShapeDeformationOrder, BlendShapeOrigin

from .core import Node
from .geometry_filter import WeightGeometryFilter


class BlendShapeNode(WeightGeometryFilter):
    """Maya blendShape node with enhanced interface."""

    node_type = "blendShape"

    def __init__(self, name: str = "blendShape") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.base_origin = Vector3Attribute(f"{self.name}.baseOrigin")
        self.deformation_order = EnumAttribute(
            f"{self.name}.deformationOrder", BlendShapeDeformationOrder
        )
        self.inbetween_info_group = ArrayAttribute(
            f"{self.name}.inbetweenInfoGroup", BlendShapeInbetweenInfoAttribute
        )
        self.input_target = ArrayAttribute(
            f"{self.name}.inputTarget", BlendShapeInputTargetAttribute
        )
        self.local_vertex_frame = BooleanAttribute(f"{self.name}.localVertexFrame")
        self.mid_layer_parent = IntegerAttribute(f"{self.name}.midLayerParent")
        self.offset_deformer = Vector3Attribute(f"{self.name}.offsetDeformer")
        self.origin = EnumAttribute(f"{self.name}.origin", BlendShapeOrigin)
        self.paint_weights = DoubleArrayAttribute(f"{self.name}.paintWeights")
        self.target_directory = ArrayAttribute(
            f"{self.name}.targetDirectory", BlendShapeTargetDirectoryAttribute
        )
        self.target_origin = Vector3Attribute(f"{self.name}.targetOrigin")
        self.topology_check = BooleanAttribute(f"{self.name}.topologyCheck")
        self.weight = ArrayAttribute(f"{self.name}.weight", ScalarAttribute)
        self.icon = ArrayAttribute(f"{self.name}.icon", StringAttribute)


class ShapeEditorManagerNode(Node):
    """Maya shapeEditorManager node with enhanced interface."""

    node_type = "shapeEditorManager"

    def __init__(self, name: str = "shapeEditorManager") -> None:
        super().__init__(name)

    def _setup_attributes(self) -> None:
        self.blend_shape_directory = ArrayAttribute(
            f"{self.name}.blendShapeDirectory", ShapeEditorManagerDirectoryAttribute
        )
        self.blend_shape_parent = ArrayAttribute(f"{self.name}.blendShapeParent", LongAttribute)
        self.out_blendshape_visibility = ArrayAttribute(
            f"{self.name}.outBlendShapeParent", BooleanAttribute
        )
