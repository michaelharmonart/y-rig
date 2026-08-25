from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

from maya import cmds

from yrig.control import Control, create_control
from yrig.joint import create_joint
from yrig.maya_api.attribute import BooleanAttribute, ClosestPointOnSurfaceResultAttribute
from yrig.maya_api.enum import RotateOrder
from yrig.maya_api.node import MultiplyNode, UvPinNode
from yrig.skin.split import tag_for_weight_split
from yrig.spline import generate_knots
from yrig.spline.curve import bound_curve_from_transforms, pin_to_curve_with_motion_path
from yrig.surface import closest_point_on_surface_reader, surface_slide_constraint, uv_pin
from yrig.transform import create_transform
from yrig.transform.utils import distance_reader

from .corner import MouthCorner


@dataclass
class LipGuides:
    lip_mid_left: str
    lip_mid: str
    lip_mid_right: str


class LipMidpoint:
    def __init__(
        self,
        name: str,
        guide: str,
        mouth_surface: str,
        corner: MouthCorner,
        control_parent: Control | str,
        parent: str,
        distance_transform: str,
        control_size: float = 1,
        mirror: bool = False,
    ):
        self.main_control = create_control(
            name,
            transform=guide,
            parent=control_parent,
            size=control_size,
            direction="z",
            rotation_order=RotateOrder.ZXY,
            limit_min_scale=False,
        )
        cmds.setAttr(f"{self.main_control.transform}.translateZ", lock=True)
        self.main_control_rest = create_transform(
            f"{name}_rest",
            parent=parent,
            transform=self.main_control.transform,
        )
        self.main_control_driven = create_transform(f"{name}_driven", parent=self.main_control_rest)
        corner_distance = distance_reader(
            corner.sub_control.offset,
            distance_transform,
            space=parent,
            zero_at_rest=True,
            axes=(True, False, False),
        )
        corner_distance_scale = MultiplyNode.create(f"{name}_distance_scale")
        corner_distance_scale.input[0].connect_from(corner_distance)
        corner_distance_scale.input[1].set(0.75)
        corner_distance_scale.output.connect_to(f"{self.main_control_driven}.translateX")

        self.main_control_slide = create_transform(f"{name}_slide", parent=parent)
        surface_slide_constraint(
            mouth_surface,
            driver_transform=self.main_control_driven,
            slider_transform=self.main_control.offset,
        )

        self.sub_control = create_control(
            f"{name}_mid_L_sub",
            transform=guide,
            parent=self.main_control,
            size=control_size * 0.5,
            direction="z",
        )
        surface_slide_constraint(
            mouth_surface,
            driver_transform=self.main_control.transform,
            slider_transform=self.sub_control.offset,
        )

        if mirror:
            cmds.setAttr(f"{self.main_control.transform}.scaleY", -1)  # type:ignore
            cmds.setAttr(f"{self.main_control.transform}.scaleX", -1)  # type:ignore
        for axis in ["X", "Y", "Z"]:
            cmds.setAttr(
                f"{self.main_control.transform}.scale{axis}",
                lock=True,
            )


class LipSpline:
    def __init__(
        self,
        name: str,
        cvs: Sequence[str],
        parent: str,
        joint_parent: str,
        surface: str,
        segments: int = 6,
        orient: bool = True,
        uv_pin_node: UvPinNode | None = None,
    ):
        degree = 3
        knots = generate_knots(len(cvs), degree, clamped=False)
        self.curve = bound_curve_from_transforms(
            cvs,
            name=name,
            parent=parent,
            degree=degree,
            knots=knots,
        )
        self.joints: list[str] = []
        self.count = segments
        self.closest_points: list[ClosestPointOnSurfaceResultAttribute] = []
        for i in range(self.count):
            segment_name = f"{self.curve}_seg{i}"
            curve_pin = create_transform(f"{segment_name}_curve_pin", parent=parent)

            pin_to_curve_with_motion_path(
                self.curve, curve_pin, parameter=(i + 0.5) / self.count, orient=orient
            )

            closest_point_reader = closest_point_on_surface_reader(curve_pin, surface)
            self.closest_points.append(closest_point_reader)

            if not orient:
                pin_slide = create_transform(f"{segment_name}_pin_slide", parent=curve_pin)
                uv_pin_node_resolved, index = uv_pin(
                    surface, pin_slide, drive_translate=False, uv_pin_node=uv_pin_node
                )
                uv_pin_node_resolved.coordinate[index].u.connect_from(
                    closest_point_reader.parameter_u
                )
                uv_pin_node_resolved.coordinate[index].v.connect_from(
                    closest_point_reader.parameter_v
                )
                driver = pin_slide
            else:
                driver = curve_pin
            joint = create_joint(
                name=segment_name, transform=driver, parent=joint_parent, radius=0.5
            )
            self.joints.append(joint)
        cmds.setAttr(f"{self.curve}.overrideEnabled", 1)  # type:ignore
        cmds.setAttr(f"{self.curve}.overrideDisplayType", 2)  # type:ignore


class Lip:
    def __init__(
        self,
        upper: bool,
        guides: LipGuides,
        mouth_surface: str,
        left_corner: MouthCorner,
        right_corner: MouthCorner,
        parent: str,
        joint_parent: str,
        control_parent: Control | str,
        control_size: float = 1,
        sub_control_vis_attr: BooleanAttribute | None = None,
        uv_pin_node: UvPinNode | None = None,
    ):
        self.guides = guides
        side_string = "upper" if upper else "lower"
        self.name = f"{side_string}_lip"
        self.group = create_transform(self.name, parent=parent)
        reference_space = str(control_parent)
        self.lip_move = create_transform(f"{self.name}_move", parent=reference_space)
        self.slider = create_transform(f"{self.name}_slide", parent=str(control_parent))
        surface_slide_constraint(
            mouth_surface, driver_transform=self.lip_move, slider_transform=self.slider
        )

        self.left_corner = left_corner
        self.right_corner = right_corner

        self.mid_left = LipMidpoint(
            name=f"{self.name}_mid_L",
            guide=guides.lip_mid_left,
            mouth_surface=mouth_surface,
            corner=self.left_corner,
            control_parent=control_parent,
            parent=self.slider,
            distance_transform=self.slider,
        )
        self.mid_control = create_control(
            f"{self.name}_mid_M",
            transform=guides.lip_mid,
            parent=self.slider,
            size=control_size,
            direction="z",
        )
        self.mid_sub_control = create_control(
            f"{self.name}_mid_M_sub",
            transform=guides.lip_mid,
            parent=self.mid_control,
            size=control_size * 0.5,
            direction="z",
        )
        surface_slide_constraint(
            mouth_surface,
            driver_transform=self.mid_control.transform,
            slider_transform=self.mid_sub_control.offset,
        )
        self.mid_right = LipMidpoint(
            name=f"{self.name}_mid_R",
            guide=guides.lip_mid_right,
            mouth_surface=mouth_surface,
            corner=self.right_corner,
            control_parent=control_parent,
            parent=self.slider,
            distance_transform=self.slider,
            mirror=True,
        )

        self.sub_controls: list[Control] = [
            self.mid_left.sub_control,
            self.mid_sub_control,
            self.mid_right.sub_control,
        ]

        if sub_control_vis_attr is not None:
            for control in self.sub_controls:
                sub_control_vis_attr.connect_to(f"{control.transform}.visibility")

        lip_cvs: tuple[Control, ...] = (
            self.mid_left.sub_control,
            self.mid_sub_control,
            self.mid_right.sub_control,
        )

        raw_left_corner_cvs: tuple[Control, ...] = (
            self.left_corner.lower_sub_control,
            self.left_corner.sub_control,
            self.left_corner.upper_sub_control,
        )
        raw_right_corner_cvs: tuple[Control, ...] = (
            self.right_corner.upper_sub_control,
            self.right_corner.sub_control,
            self.right_corner.lower_sub_control,
        )

        if upper:
            left_corner_cvs = raw_left_corner_cvs
            right_corner_cvs = raw_right_corner_cvs
        else:
            # Reverse order of corner controls for lower lip
            left_corner_cvs = raw_left_corner_cvs[::-1]
            right_corner_cvs = raw_right_corner_cvs[::-1]

        full_lip_cvs = left_corner_cvs + lip_cvs + right_corner_cvs
        left_lip_cvs = left_corner_cvs + lip_cvs
        right_lip_cvs = right_corner_cvs[::-1] + lip_cvs[::-1]

        self.main_joint = create_joint(name=f"{self.name}_main", parent=joint_parent)

        self.left_main_spline = LipSpline(
            f"{self.name}_L_main_spline",
            [control.offset for control in left_lip_cvs],
            parent=self.group,
            joint_parent=self.main_joint,
            surface=mouth_surface,
            orient=False,
            uv_pin_node=uv_pin_node,
        )
        self.right_main_spline = LipSpline(
            f"{self.name}_R_main_spline",
            [control.offset for control in right_lip_cvs],
            parent=self.group,
            joint_parent=self.main_joint,
            surface=mouth_surface,
            orient=False,
            uv_pin_node=uv_pin_node,
        )
        tag_for_weight_split(
            self.main_joint,
            chain(self.left_main_spline.joints, reversed(self.right_main_spline.joints)),
        )

        self.sub_joint = create_joint(name=f"{self.name}_sub", parent=joint_parent)

        self.left_sub_spline = LipSpline(
            f"{self.name}_L_sub_spline",
            [control.transform for control in left_lip_cvs],
            parent=self.group,
            joint_parent=self.sub_joint,
            surface=mouth_surface,
        )
        self.right_sub_spline = LipSpline(
            f"{self.name}_R_sub_spline",
            [control.transform for control in right_lip_cvs],
            parent=self.group,
            joint_parent=self.sub_joint,
            surface=mouth_surface,
        )
        tag_for_weight_split(
            self.sub_joint,
            chain(self.left_sub_spline.joints, reversed(self.right_sub_spline.joints)),
        )
