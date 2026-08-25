from dataclasses import dataclass, field

from maya import cmds

from yrig.control import Control, ControlShape, create_control
from yrig.joint import create_joint
from yrig.maya_api.enum import RotateOrder
from yrig.maya_api.node import UvPinNode
from yrig.surface import get_surface_shapes, surface_slide_constraint
from yrig.transform import create_transform, matrix_constraint
from yrig.transform.matrix import local_constraint

from .cheek import CheekInterpolate, CheekInterpolateGuides
from .corner import MouthCorner
from .lip import Lip, LipGuides


def _default_lip_guides(side: str) -> LipGuides:
    return LipGuides(
        lip_mid_left=f"{side}_lip_mid_L",
        lip_mid=f"{side}_lip_mid_M",
        lip_mid_right=f"{side}_lip_mid_R",
    )


@dataclass
class MouthGuides:
    mouth: str = "mouth_M"
    mouth_surface: str = "face_surface"
    jaw: str = "jaw_M"
    left_corner: str = "mouth_corner_L"
    right_corner: str = "mouth_corner_R"
    upper_lip: LipGuides = field(default_factory=lambda: _default_lip_guides(side="upper"))
    lower_lip: LipGuides = field(default_factory=lambda: _default_lip_guides(side="lower"))
    cheek_interpolate: CheekInterpolateGuides = field(default_factory=CheekInterpolateGuides)


class Mouth:
    def __init__(
        self,
        guides: MouthGuides,
        parent: str,
        control_parent: Control | str,
        joint_parent: str,
        control_size: float = 1,
    ):
        self.guides = guides

        duplicated_mouth_surface = cmds.duplicate(self.guides.mouth_surface)[0]
        cmds.parent(duplicated_mouth_surface, parent)
        self.mouth_surface = cmds.rename(duplicated_mouth_surface, "mouth_surface")
        cmds.hide(self.mouth_surface)

        reference_space = str(control_parent)

        self.mouth_control = create_control(
            "mouth_M",
            transform=guides.mouth,
            parent=control_parent,
            size=control_size * 5,
            control_shape=ControlShape.LINE,
            direction="z",
        )
        self.jaw_control = create_control(
            "jaw_M",
            transform=guides.jaw,
            parent=control_parent,
            size=control_size * 8,
            control_shape=ControlShape.LINE,
            direction="z",
            rotation_order=RotateOrder.YZX,
        )
        self.jaw_joint = create_joint(name="jaw_M", transform=self.jaw_control, parent=joint_parent)
        self.jaw_blend = create_transform(
            "jaw_M_blend", parent=parent, transform=self.jaw_control.transform
        )
        self.mouth_slide = create_transform("mouth_M_slide", parent=self.mouth_control.transform)
        surface_slide_constraint(
            self.mouth_surface,
            driver_transform=self.mouth_control.transform,
            slider_transform=self.mouth_slide,
        )
        cmds.parentConstraint(
            self.jaw_control.transform,
            reference_space,
            self.jaw_blend,
            maintainOffset=True,
        )
        self.jaw_blend_local = create_transform(
            "jaw_M_blend_local", parent=self.mouth_slide, transform=self.jaw_control.transform
        )
        local_constraint(self.jaw_blend, self.jaw_blend_local, reference_space=reference_space)

        self.left_corner = MouthCorner(
            side="L",
            guide=guides.left_corner,
            mouth_surface=self.mouth_surface,
            control_parent=self.mouth_slide,
            control_size=control_size,
        )
        self.right_corner = MouthCorner(
            side="R",
            guide=guides.right_corner,
            mouth_surface=self.mouth_surface,
            control_parent=self.mouth_slide,
            control_size=control_size,
        )

        matrix_constraint(self.jaw_blend_local, self.left_corner.main_control.offset)
        matrix_constraint(self.jaw_blend_local, self.right_corner.main_control.offset)

        mouth_uv_pin = UvPinNode.create("mouth_surface_uvPin")
        primary_shape, original_shape, shape_output = get_surface_shapes(self.mouth_surface)
        mouth_uv_pin.original_geometry.connect_from(f"{original_shape}.{shape_output}")
        mouth_uv_pin.deformed_geometry.connect_from(f"{primary_shape}.{shape_output}")

        self.upper_lip = Lip(
            upper=True,
            guides=guides.upper_lip,
            mouth_surface=self.mouth_surface,
            left_corner=self.left_corner,
            right_corner=self.right_corner,
            parent=parent,
            joint_parent=joint_parent,
            control_parent=self.mouth_slide,
            control_size=control_size,
            uv_pin_node=mouth_uv_pin,
        )
        self.lower_lip = Lip(
            upper=False,
            guides=guides.lower_lip,
            mouth_surface=self.mouth_surface,
            left_corner=self.left_corner,
            right_corner=self.right_corner,
            parent=parent,
            joint_parent=joint_parent,
            control_parent=self.mouth_slide,
            control_size=control_size,
            uv_pin_node=mouth_uv_pin,
        )

        self.cheek_interpolate = CheekInterpolate(
            guides=self.guides.cheek_interpolate,
            mouth_surface=self.mouth_surface,
            upper_left_lip_spline=self.upper_lip.left_main_spline,
            upper_right_lip_spline=self.upper_lip.right_main_spline,
            lower_left_lip_spline=self.lower_lip.left_main_spline,
            lower_right_lip_spline=self.lower_lip.right_main_spline,
            parent=parent,
            joint_parent=joint_parent,
            control_parent=self.mouth_slide,
            control_size=control_size,
        )

        local_constraint(
            self.jaw_control.transform, self.lower_lip.lip_move, reference_space=reference_space
        )
