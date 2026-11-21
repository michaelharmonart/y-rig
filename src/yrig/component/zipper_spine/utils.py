from maya import cmds

from yrig.transform import get_position

from .guide.core import GuideInfo, create_guide_from_position


def curve_midpoint(curve: str) -> GuideInfo:
    pos = cmds.pointOnCurve(
        curve,
        parameter=0.5,
        position=True,
        turnOnPercentage=True,
    )

    mid_guide = create_guide_from_position(guide_name="mid_point", parent="guides", pos=tuple(pos))  # type:ignore

    return mid_guide


def align_guides(guide_01: GuideInfo, guide_02: GuideInfo, flip_y: bool = False) -> None:
    if guide_01.pos[2] > guide_02.pos[2]:
        mod = -1
    else:
        mod = 1
    ymod = -1 if flip_y else 1

    aim = cmds.aimConstraint(
        guide_02.name,
        guide_01.name,
        aimVector=(0, 1 * ymod, 0),  # Primary / aim axis = +X
        upVector=(0, 0, -1 * mod),  # Up axis = +Y
        worldUpType="vector",
        worldUpVector=(0, 1, 0),
        maintainOffset=False,
    )

    cmds.delete(aim)  # type:ignore


def get_distance_between(obj_a: str, obj_b: str, world_space: bool = True) -> float:
    p1 = get_position(obj_a, world_space)
    p2 = get_position(obj_b, world_space)

    return p1.distanceTo(p2)
