from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

from yrig.control import Control
from yrig.joint import create_joint
from yrig.maya_api.attribute import BooleanAttribute, UvPinCoordinateAttribute
from yrig.maya_api.node import BlendColorsNode, UvPinNode
from yrig.skin.split import tag_for_weight_split
from yrig.spline import generate_knots, get_point_on_spline, resample
from yrig.structs.transform import Vector3
from yrig.surface import uv_pin, uv_pin_multi
from yrig.transform import create_transform
from yrig.transform.utils import get_position, set_position

from .lip import LipSpline


@dataclass
class CheekInterpolateGuides:
    max_upper_mid_cv: str = "mouth_interpolate_max_M_upper_jnt"
    max_lower_mid_cv: str = "mouth_interpolate_max_M_lower_jnt"
    max_left_corner: str = "mouth_interpolate_max_L_corner_jnt"
    max_right_corner: str = "mouth_interpolate_max_R_corner_jnt"
    max_upper_left_cvs: Sequence[str] = (
        "mouth_interpolate_max_L_upper_cv0_jnt",
        "mouth_interpolate_max_L_upper_cv1_jnt",
    )
    max_upper_right_cvs: Sequence[str] = (
        "mouth_interpolate_max_R_upper_cv0_jnt",
        "mouth_interpolate_max_R_upper_cv1_jnt",
    )
    max_lower_left_cvs: Sequence[str] = (
        "mouth_interpolate_max_L_lower_cv0_jnt",
        "mouth_interpolate_max_L_lower_cv1_jnt",
    )
    max_lower_right_cvs: Sequence[str] = (
        "mouth_interpolate_max_R_lower_cv0_jnt",
        "mouth_interpolate_max_R_lower_cv1_jnt",
    )


class CheekInterpolateMax:
    def __init__(
        self,
        name: str,
        cvs: Sequence[str],
        parent: str,
        surface: str,
        uv_pin_node: UvPinNode,
        segments: int = 6,
    ):
        cv_positions: list[Vector3] = []

        for transform in cvs:
            transform_position = get_position(transform)
            cv_positions.append(
                Vector3(transform_position.x, transform_position.y, transform_position.z)
            )
        knots = generate_knots(len(cvs), clamped=False)
        parameters = resample(
            cv_positions, number_of_points=segments, knots=knots, normalize_parameter=False
        )

        self.pinned_transforms: list[str] = []
        self.pinned_uv_attrs: list[UvPinCoordinateAttribute] = []
        for index, parameter in enumerate(parameters):
            position = get_point_on_spline(
                cv_positions=cv_positions, t=parameter, knots=knots, normalize_parameter=False
            )
            transform = create_transform(f"{name}_seg{index}", parent=parent)
            set_position(transform, (position.x, position.y, position.z))
            uv_pin_node, index = uv_pin(surface, transform, uv_pin_node=uv_pin_node)
            self.pinned_transforms.append(transform)
            self.pinned_uv_attrs.append(uv_pin_node.coordinate[index])


class CheekInterpolateMid:
    def __init__(
        self,
        name: str,
        parent: str,
        joint_parent: str,
        surface: str,
        uv_pin_node: UvPinNode,
        max: CheekInterpolateMax,
        lip_spline: LipSpline,
        blend: float = 0.5,
    ):
        self.joints: list[str] = []
        for index, (max_point, lip_point) in enumerate(
            zip(max.pinned_uv_attrs, lip_spline.closest_points, strict=True)
        ):
            segment_name = f"{name}_seg{index}"
            joint = create_joint(segment_name, parent=joint_parent)
            self.joints.append(joint)
            _, index = uv_pin(surface, joint, uv_pin_node=uv_pin_node)
            blend_node = BlendColorsNode.create(f"{segment_name}_blend")
            blend_node.color1.r.connect_from(lip_point.parameter_u)
            blend_node.color1.g.connect_from(lip_point.parameter_v)
            blend_node.color2.r.set(max_point.u.get())
            blend_node.color2.g.set(max_point.v.get())
            blend_node.blender.set(blend)
            uv_pin_node.coordinate[index].u.connect_from(blend_node.output.r)
            uv_pin_node.coordinate[index].v.connect_from(blend_node.output.g)


class CheekInterpolateLoop:
    def __init__(
        self,
        name: str,
        max_splines: tuple[
            CheekInterpolateMax, CheekInterpolateMax, CheekInterpolateMax, CheekInterpolateMax
        ],
        lip_splines: tuple[LipSpline, LipSpline, LipSpline, LipSpline],
        mouth_surface: str,
        parent: str,
        joint_parent: str,
        uv_pin_node: UvPinNode,
        blend: float = 0.5,
        tag_for_split: bool = True,
    ):
        self.joint = create_joint(name=name, parent=joint_parent)
        self.upper_left = CheekInterpolateMid(
            name=f"{name}_upper_L",
            parent=parent,
            joint_parent=self.joint,
            surface=mouth_surface,
            uv_pin_node=uv_pin_node,
            max=max_splines[0],
            lip_spline=lip_splines[0],
            blend=blend,
        )
        self.upper_right = CheekInterpolateMid(
            name=f"{name}_upper_R",
            parent=parent,
            joint_parent=self.joint,
            surface=mouth_surface,
            uv_pin_node=uv_pin_node,
            max=max_splines[1],
            lip_spline=lip_splines[1],
            blend=blend,
        )
        self.lower_left = CheekInterpolateMid(
            name=f"{name}_lower_L",
            parent=parent,
            joint_parent=self.joint,
            surface=mouth_surface,
            uv_pin_node=uv_pin_node,
            max=max_splines[2],
            lip_spline=lip_splines[2],
            blend=blend,
        )
        self.lower_right = CheekInterpolateMid(
            name=f"{name}_lower_R",
            parent=parent,
            joint_parent=self.joint,
            surface=mouth_surface,
            uv_pin_node=uv_pin_node,
            max=max_splines[3],
            lip_spline=lip_splines[3],
            blend=blend,
        )

        if tag_for_split:
            tag_for_weight_split(
                self.joint,
                chain(
                    self.upper_left.joints,
                    reversed(self.upper_right.joints),
                    self.lower_right.joints,
                    reversed(self.lower_left.joints),
                ),
            )


def _create_cv_transforms(name_format: str, cv_guides: Sequence[str], parent: str) -> list[str]:
    transforms: list[str] = []
    for index, cv_guide in enumerate(cv_guides):
        transform = create_transform(f"{name_format}{index}", transform=cv_guide, parent=parent)
        transforms.append(transform)
    return transforms


class CheekInterpolate:
    def __init__(
        self,
        guides: CheekInterpolateGuides,
        mouth_surface: str,
        upper_left_lip_spline: LipSpline,
        upper_right_lip_spline: LipSpline,
        lower_left_lip_spline: LipSpline,
        lower_right_lip_spline: LipSpline,
        parent: str,
        control_parent: Control | str,
        joint_parent: str,
        control_size: float = 1,
        sub_control_vis_attr: BooleanAttribute | None = None,
        tag_for_split: bool = True,
    ):
        self.guides = guides
        self.name = "cheek_interpolate"
        self.group = create_transform(f"{self.name}_grp", parent=parent)

        cheek_max_name = "cheek_max_interp"

        self.max_upper_mid_cv = create_transform(
            f"{cheek_max_name}_upper_M", transform=self.guides.max_upper_mid_cv, parent=self.group
        )
        self.max_lower_mid_cv = create_transform(
            f"{cheek_max_name}_lower_M", transform=self.guides.max_lower_mid_cv, parent=self.group
        )
        self.max_left_corner_cv = create_transform(
            f"{cheek_max_name}_corner_L", transform=self.guides.max_left_corner, parent=self.group
        )
        self.max_right_corner_cv = create_transform(
            f"{cheek_max_name}_corner_R", transform=self.guides.max_right_corner, parent=self.group
        )
        self.max_upper_left_cvs = _create_cv_transforms(
            f"{cheek_max_name}_upper_L_cv",
            cv_guides=self.guides.max_upper_left_cvs,
            parent=self.group,
        )
        self.max_upper_right_cvs = _create_cv_transforms(
            f"{cheek_max_name}_upper_R_cv",
            cv_guides=self.guides.max_upper_right_cvs,
            parent=self.group,
        )
        self.max_lower_left_cvs = _create_cv_transforms(
            f"{cheek_max_name}_lower_L_cv",
            cv_guides=self.guides.max_lower_left_cvs,
            parent=self.group,
        )
        self.max_lower_right_cvs = _create_cv_transforms(
            f"{cheek_max_name}_lower_R_cv",
            cv_guides=self.guides.max_lower_right_cvs,
            parent=self.group,
        )
        self.cv_transforms = (
            [self.max_upper_mid_cv]
            + [self.max_lower_mid_cv]
            + [self.max_left_corner_cv]
            + [self.max_right_corner_cv]
            + self.max_upper_left_cvs
            + self.max_upper_right_cvs
            + self.max_lower_left_cvs
            + self.max_lower_right_cvs
        )

        self.max_upper_left_full_cvs = (
            [self.max_lower_left_cvs[-1]]
            + [self.max_left_corner_cv]
            + self.max_upper_left_cvs
            + [self.max_upper_mid_cv]
            + [self.max_upper_right_cvs[0]]
        )
        self.max_upper_right_full_cvs = (
            [self.max_lower_right_cvs[-1]]
            + [self.max_right_corner_cv]
            + self.max_upper_right_cvs
            + [self.max_upper_mid_cv]
            + [self.max_upper_left_cvs[0]]
        )

        self.max_lower_left_full_cvs = (
            [self.max_upper_left_cvs[-1]]
            + [self.max_left_corner_cv]
            + self.max_lower_left_cvs
            + [self.max_lower_mid_cv]
            + [self.max_lower_right_cvs[0]]
        )
        self.max_lower_right_full_cvs = (
            [self.max_upper_right_cvs[-1]]
            + [self.max_right_corner_cv]
            + self.max_lower_right_cvs
            + [self.max_lower_mid_cv]
            + [self.max_lower_left_cvs[0]]
        )

        self.uv_pin = uv_pin_multi(
            "cheek_interpolate_uvPin", mouth_surface, self.cv_transforms, keep_offset=True
        )

        cheek_max_name = "cheek_interpolate_max"
        self.upper_left_max = CheekInterpolateMax(
            name=f"{cheek_max_name}_upper_L",
            cvs=self.max_upper_left_full_cvs,
            parent=self.group,
            surface=mouth_surface,
            uv_pin_node=self.uv_pin,
            segments=upper_left_lip_spline.count,
        )
        self.upper_right_max = CheekInterpolateMax(
            name=f"{cheek_max_name}_upper_R",
            cvs=self.max_upper_right_full_cvs,
            parent=self.group,
            surface=mouth_surface,
            uv_pin_node=self.uv_pin,
            segments=upper_right_lip_spline.count,
        )
        self.lower_left_max = CheekInterpolateMax(
            name=f"{cheek_max_name}_lower_L",
            cvs=self.max_lower_left_full_cvs,
            parent=self.group,
            surface=mouth_surface,
            uv_pin_node=self.uv_pin,
            segments=lower_left_lip_spline.count,
        )
        self.lower_right_max = CheekInterpolateMax(
            name=f"{cheek_max_name}_lower_R",
            cvs=self.max_lower_right_full_cvs,
            parent=self.group,
            surface=mouth_surface,
            uv_pin_node=self.uv_pin,
            segments=lower_right_lip_spline.count,
        )

        self.outer_mid_loop = CheekInterpolateLoop(
            name="cheek_interpolate_outer",
            parent=self.group,
            joint_parent=joint_parent,
            mouth_surface=mouth_surface,
            uv_pin_node=self.uv_pin,
            max_splines=(
                self.upper_left_max,
                self.upper_right_max,
                self.lower_left_max,
                self.lower_right_max,
            ),
            lip_splines=(
                upper_left_lip_spline,
                upper_right_lip_spline,
                lower_left_lip_spline,
                lower_right_lip_spline,
            ),
            tag_for_split=tag_for_split,
            blend=1 / 3,
        )

        self.inner_mid_loop = CheekInterpolateLoop(
            name="cheek_interpolate_inner",
            parent=self.group,
            joint_parent=joint_parent,
            mouth_surface=mouth_surface,
            uv_pin_node=self.uv_pin,
            max_splines=(
                self.upper_left_max,
                self.upper_right_max,
                self.lower_left_max,
                self.lower_right_max,
            ),
            lip_splines=(
                upper_left_lip_spline,
                upper_right_lip_spline,
                lower_left_lip_spline,
                lower_right_lip_spline,
            ),
            tag_for_split=tag_for_split,
            blend=2 / 3,
        )
