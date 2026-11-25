import maya.api.OpenMaya as om
from maya import cmds

from yrig.transform import get_position

from .guide.core import GuideInfo, create_guide_from_position


def curve_midpoint(curve: str) -> GuideInfo:
    pos = cmds.pointOnCurve(curve, parameter=0.5, position=True, turnOnPercentage=True)
    return create_guide_from_position(guide_name="mid_point", parent="guides", pos=tuple(pos))  # type: ignore


def align_guides(
    guide_01: GuideInfo,
    guide_02: GuideInfo,
    flip_x: bool = False,
) -> None:

    aim = cmds.aimConstraint(
        guide_02.name,
        guide_01.name,
        aimVector=(-1, 0, 0) if flip_x else (1, 0, 0),
        upVector=(0, 0, 1),
        worldUpType="vector",
        worldUpVector=(0, 0, 1),
        maintainOffset=False,
    )

    cmds.delete(aim)  # type: ignore


def get_distance_between(obj_a: str, obj_b: str, world_space: bool = True) -> float:
    return get_position(obj_a, world_space).distanceTo(get_position(obj_b, world_space))


def get_closest_curve_matrix(curve: str, transform: str) -> list[float]:
    selection = om.MSelectionList()
    selection.add(curve)
    dag_path = selection.getDagPath(0)
    dag_path.extendToShape()
    curve_fn = om.MFnNurbsCurve(dag_path)
    position = cmds.xform(transform, query=True, worldSpace=True, translation=True)
    point = om.MPoint(*position)  # type: ignore
    closest_point, _ = curve_fn.closestPoint(point, space=om.MSpace.kWorld)
    matrix = cmds.xform(transform, query=True, worldSpace=True, matrix=True)
    matrix[12:15] = [closest_point.x, closest_point.y, closest_point.z]  # type: ignore
    return matrix  # type: ignore
