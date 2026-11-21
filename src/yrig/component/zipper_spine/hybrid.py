from dataclasses import dataclass

from maya import cmds

from yrig.control.core import Control, create_control
from yrig.joint import create_joint
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.transform.utils import create_transform

from .guide.curve import GuideCurve, get_curve_point_matrix
from .utils import curve_midpoint


@dataclass
class HybridInfo:
    mid: Control
    sub: list
    joints: list
    grp: str


def build_hybrid(
    part: str,
    guts: str,
    side: str,
    control_grp: str,
    guide_curve: GuideCurve,
    guide_list: list,
    cv_transforms: list,
    rig_size: float,
    hip: Control,
    hip_des: str,
    main_color: str,
    mid_control_size: float = 1 / 3,
    sub_spine_control_size: float = 1 / 30,
) -> HybridInfo:
    hybrid_grp = create_transform(name=f"{part}_{side}_hybrid_grp", parent=control_grp)

    mid_guide = curve_midpoint(curve=guide_curve.curve)

    mid_hip = create_transform(
        name=f"{hip_des}_mid_{side}_grp", parent=hybrid_grp, transform=hip.transform
    )
    mid = create_control(
        name=f"{part}_mid_{side}",
        parent=mid_hip,
        transform=mid_guide.name,
        size=rig_size * mid_control_size,
        control_shape="circle",
        direction="y",
    )

    lower_matrix = get_curve_point_matrix(
        curve=guide_curve.curve,
        parameter=0.15,
        tangent_orient=True,
    )

    upper_matrix = get_curve_point_matrix(
        curve=guide_curve.curve,
        parameter=0.85,
        tangent_orient=True,
    )

    lower_driver = create_transform(
        name=f"{part}_lower_driver", parent=cv_transforms[0], transform=lower_matrix
    )
    upper_driver = create_transform(
        name=f"{part}_upper_driver", parent=cv_transforms[-1], transform=upper_matrix
    )
    for attrs in ["rotateX", "rotateY", "rotateZ"]:
        cmds.setAttr(f"{lower_driver}.{attrs}", 0)  # type:ignore
        cmds.setAttr(f"{upper_driver}.{attrs}", 0)  # type:ignore

    cv_transforms.insert(1, lower_driver)
    cv_transforms.insert(2, mid.transform)
    cv_transforms.insert(3, upper_driver)

    spine_off_ctrls = []
    spine_driven = []
    jnts = []

    jnt_par = guts
    for g in guide_list:
        ctrl = create_control(
            name=g.descriptor,
            parent=hybrid_grp,
            transform=g.name,
            size=rig_size * sub_spine_control_size,
            control_shape="circle",
            direction="z",
        )
        jnt = create_joint(
            name=f"Hybrid_{g.descriptor}",
            transform=ctrl.transform,
            parent=jnt_par,
        )
        spine_off_ctrls.append(ctrl)
        spine_driven.append(ctrl.offset)
        jnt_par = jnt
        jnts.append(jnt)

    matrix_spline_from_transforms(
        name=f"{part}_{side}_ms",
        pinned_transforms=spine_driven,
        cv_transforms=cv_transforms,
        parent=guts,
        degree=2,
    )

    return HybridInfo(mid=mid, sub=spine_off_ctrls, joints=jnts, grp=hybrid_grp)
