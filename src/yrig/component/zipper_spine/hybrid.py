from dataclasses import dataclass

from maya import cmds

from yrig.control import set_override_color
from yrig.control.core import Control, create_control
from yrig.joint import create_joint
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.transform.utils import create_transform

from .guide.curve import GuideCurve, get_curve_point_matrix
from .stabilization import (
    build_stabilized_drivers,
    connect_limited_control,
    create_limited_control,
    create_stretch_attributes,
    limit_spine_length,
    publish_spine_measurements,
)


@dataclass
class HybridInfo:
    mid: Control
    sub: list[Control]
    joints: list[str]
    grp: str
    drivers: list[str]
    chest_driver: str
    min_stretch: str
    max_stretch: str
    soft_range: str


def build_hybrid(
    part: str,
    guts: str,
    side: str,
    control_grp: str,
    guide_curve: GuideCurve,
    guide_list: list,
    cv_transforms: list[str],
    rig_size: float,
    hip: Control,
    hip_des: str,
    main_color: str,
    cog: str,
    chest: str,
    mid_control_size: float = 1 / 3,
    sub_spine_control_size: float = 1 / 30,
    soft_influence: float = 0.2,
    longitudinal_influence: float = 0.05,
    compression_min: float = 0.25,
    compression_max: float = 0.75,
    max_offset_ratio: float = 0.3,
    min_stretch: float = 0.5,
    max_stretch: float = 1.5,
    soft_range: float = 0.05,
) -> HybridInfo:

    hybrid_grp = create_transform(
        name=f"{part}_{side}_hybrid_grp",
        parent=control_grp,
    )

    stretch_attrs = create_stretch_attributes(
        hybrid_grp=hybrid_grp,
        min_stretch=min_stretch,
        max_stretch=max_stretch,
        soft_range=soft_range,
    )

    if len(cv_transforms) != 2:
        raise ValueError("Expected hip and chest endpoint drivers.")

    # -----------------------------------
    # Mid control - Y up
    # -----------------------------------

    mid_hip = create_transform(
        name=f"{hip_des}_mid_{side}_grp",
        parent=hybrid_grp,
        transform=hip.transform,
    )

    mid_position = cmds.pointOnCurve(
        guide_curve.curve,
        parameter=0.5,
        turnOnPercentage=True,
        position=True,
    )

    mid_guide = create_transform(
        name=f"{part}_mid_{side}_guide",
        parent=mid_hip,
        transform=hip.transform,
    )

    cmds.xform(
        mid_guide,
        worldSpace=True,
        translation=mid_position,  # type:ignore
    )

    mid = create_control(
        name=f"{part}_mid_{side}",
        parent=mid_hip,
        transform=mid_guide,
        size=rig_size * mid_control_size,
        control_shape="circle",
        direction="y",
    )

    cmds.delete(mid_guide)

    set_override_color(
        control=mid,
        color=(0.957, 0.725, 0.259),
    )

    mid_input = create_limited_control(
        control=mid,
        follow_driver=chest,
        parent=hybrid_grp,
    )

    # -----------------------------------
    # Lower and upper drivers - Y up
    # -----------------------------------

    lower_matrix = get_curve_point_matrix(
        curve=guide_curve.curve,
        parameter=0.15,
        tangent_orient=True,
        aim_axis="y",
    )

    upper_matrix = get_curve_point_matrix(
        curve=guide_curve.curve,
        parameter=0.85,
        tangent_orient=True,
        aim_axis="y",
    )

    lower_driver = create_transform(
        name=f"{part}_{side}_lower_driver",
        parent=cv_transforms[0],
        transform=lower_matrix,
    )

    upper_driver = create_transform(
        name=f"{part}_{side}_upper_driver",
        parent=cv_transforms[-1],
        transform=upper_matrix,
    )

    for driver in (lower_driver, upper_driver):
        for axis in "XYZ":
            cmds.setAttr(f"{driver}.rotate{axis}", 0)  # type:ignore

    # -----------------------------------
    # Raw spline drivers
    # -----------------------------------

    raw_drivers = [
        cv_transforms[0],
        lower_driver,
        mid_input.raw_driver,
        upper_driver,
        cv_transforms[-1],
    ]

    # -----------------------------------
    # Stabilized spline drivers
    # -----------------------------------

    stable_drivers = build_stabilized_drivers(
        name=f"{part}_{side}_hybrid",
        drivers=raw_drivers,
        cog=cog,
        soft_influence=soft_influence,
        longitudinal_influence=longitudinal_influence,
        compression_min=compression_min,
        compression_max=compression_max,
        max_offset_ratio=max_offset_ratio,
    )

    # -----------------------------------
    # Squash and stretch limits
    # -----------------------------------

    limited_drivers = limit_spine_length(
        name=f"{part}_{side}_hybrid",
        drivers=stable_drivers,
        cog=cog,
        hybrid_grp=hybrid_grp,
        min_stretch=stretch_attrs["min_stretch"],
        max_stretch=stretch_attrs["max_stretch"],
        soft_range=stretch_attrs["soft_range"],
    )

    connect_limited_control(
        control=mid_input,
        limited_driver=limited_drivers[2],
        name=f"{part}_{side}_mid",
    )

    # -----------------------------------
    # Sub controls and joints - X down chain
    # -----------------------------------

    spine_off_ctrls: list[Control] = []
    spine_driven: list[str] = []
    jnts: list[str] = []

    jnt_par = guts

    for guide in guide_list:
        ctrl = create_control(
            name=guide.descriptor,
            parent=hybrid_grp,
            transform=guide.name,
            size=rig_size * sub_spine_control_size,
            control_shape="circle",
            direction="x",
        )

        set_override_color(
            control=ctrl,
            color=(0.718, 0.518, 0.655),
        )

        jnt = create_joint(
            name=f"Hybrid_{guide.descriptor}",
            transform=ctrl.transform,
            parent=jnt_par,
        )

        spine_off_ctrls.append(ctrl)
        spine_driven.append(ctrl.offset)

        jnt_par = jnt
        jnts.append(jnt)

    # -----------------------------------
    # Matrix spline - X down chain
    # -----------------------------------

    matrix_spline_from_transforms(
        name=f"{part}_{side}_ms",
        pinned_transforms=spine_driven,
        cv_transforms=limited_drivers,
        parent=guts,
        degree=2,
        # Output orientation
        primary_axis=(1, 0, 0),
        secondary_axis=(0, 0, 1),
        # Orientation behavior
        align_tangent=True,
        interpolate_rotation=True,
        twist=True,
        # Disable automatic scaling
        stretch=False,
        interpolate_scale=False,
    )

    # -----------------------------------
    # Squash and stretch measurements
    # -----------------------------------

    publish_spine_measurements(
        name=f"{part}_{side}_hybrid",
        hybrid_grp=hybrid_grp,
        drivers=limited_drivers,
        spline_pins=spine_driven,
        cog=cog,
    )

    return HybridInfo(
        mid=mid,
        sub=spine_off_ctrls,
        joints=jnts,
        grp=hybrid_grp,
        drivers=limited_drivers,
        chest_driver=limited_drivers[-1],
        min_stretch=stretch_attrs["min_stretch"],
        max_stretch=stretch_attrs["max_stretch"],
        soft_range=stretch_attrs["soft_range"],
    )
