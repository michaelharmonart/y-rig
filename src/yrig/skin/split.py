from typing import Sequence, TypeVar

import maya.cmds as cmds
from maya.api import OpenMaya as om2
from maya.api.OpenMaya import (
    MDagPath,
    MFnNurbsCurve,
    MFnNurbsCurveData,
    MFnNurbsSurface,
    MObject,
    MPoint,
    MPointArray,
    MSelectionList,
)

from yrig import spline
from yrig.math import remap
from yrig.skin.core import get_mesh_points

# CV can be anything: a Vector3, a transform name, etc.
CV = TypeVar("CV")


def get_mesh_spline_weights(
    mesh_shape: str,
    cv_transforms: Sequence[str],
    degree: int = 2,
    periodic: bool = False,
    vertex_indices: list[int] | None = None,
    debug_curve: bool = False,
) -> list[list[tuple[str, float]]]:
    """
    Calculates spline-based weights for each vertex on a mesh relative to a temporary NURBS curve
    defined by a set of CV transforms.

    The function builds a curve from the given transforms, projects each mesh vertex onto the curve
    to compute the closest parameter value, then calculates De Boor-style basis weights using the
    curve's knot vector and degree.

    Args:
        mesh_shape (str): The name of the mesh shape node (not the transform).
        cv_transforms (list[str]): A list of transform names representing the CVs of the curve.
        degree (int, optional): Degree of the spline curve. Defaults to 2.
        periodic: If True will generate a periodic curve for getting spline weights.
        vertex_indices: A list of vertex indices to output weights for.
        debug_curve: If True a curve node will be created for debug purposes.
    Returns:
        list[list[tuple[Any, float]]]: A list of weights per vertex. Each entry is a list of tuples,
        where each tuple contains a CV transform and its corresponding influence weight on the vertex.
    """
    # Create a curve for checking the closest point
    cv_positions: MPointArray = MPointArray()
    for transform in cv_transforms:
        position: tuple[float, float, float] = tuple(
            cmds.xform(  # type: ignore
                transform, query=True, worldSpace=True, translation=True
            )
        )
        cv_positions.append(MPoint(*position))

    extended_cv_positions: MPointArray
    extended_cv_transforms: list[str]
    if periodic:
        extended_cv_positions = MPointArray(cv_positions) + cv_positions[:degree]
        extended_cv_transforms = list(cv_transforms) + list(cv_transforms)[:degree]
    else:
        extended_cv_positions = MPointArray(cv_positions)
        extended_cv_transforms = list(cv_transforms)
    knots: list[float] = spline.generate_knots(
        len(extended_cv_positions), degree=degree, periodic=periodic
    )
    maya_knots: list[float] = knots[1:-1]

    fn_data: MFnNurbsCurveData = om2.MFnNurbsCurveData()
    data_obj: MObject = fn_data.create()
    fn_curve: MFnNurbsCurve = om2.MFnNurbsCurve()
    fn_curve.create(
        extended_cv_positions,
        om2.MDoubleArray(maya_knots),
        degree,
        om2.MFnNurbsCurve.kOpen if not periodic else om2.MFnNurbsCurve.kPeriodic,
        False,  # create2D
        False,  # not rational
        data_obj,
    )

    if debug_curve:
        cmds.curve(
            name=f"{mesh_shape}_SplineWeightsDebugCurve",
            point=[
                (cv_position.x, cv_position.y, cv_position.z)
                for cv_position in extended_cv_positions
            ],
            periodic=periodic,
            knot=maya_knots,
            degree=degree,
            worldSpace=True,
        )

    # get the MDagPaths
    msel: om2.MSelectionList = om2.MSelectionList()
    msel.add(mesh_shape)
    mesh_dag: om2.MDagPath = msel.getDagPath(0)

    # make the function set and get the points
    fn_mesh: om2.MFnMesh = om2.MFnMesh(mesh_dag)

    # get the points in world space

    mesh_points: MPointArray = get_mesh_points(fn_mesh=fn_mesh, vertex_indices=vertex_indices)

    # iterate over the points and get the closest parameter
    parameters: list[float] = []
    for i, point in enumerate(mesh_points):  # type: ignore
        parameter: float = fn_curve.closestPoint(point, space=om2.MSpace.kObject)[1]
        parameters.append(parameter)

    spline_weights_per_vertex: list[list[tuple[str, float]]] = spline.get_weights_along_spline(
        cvs=extended_cv_transforms, parameters=parameters, degree=degree, knots=knots
    )

    return spline_weights_per_vertex


def get_mesh_surface_weights(
    mesh_shape: str,
    surface_shape: str,
    influence_transforms: Sequence[CV],
    degree: int = 2,
    vertex_indices: list[int] | None = None,
) -> list[list[tuple[CV, float]]]:
    """
    Calculates weights for each vertex on a mesh relative to a given NURBS surface.

    The function projects each mesh vertex onto the surface to compute the closest parameter value,
    then calculates De Boor basis weights using the parameter.

    Args:
        mesh_shape (str): The name of the mesh shape node (not the transform).
        surface_shape (str): The name of the NUBRS surface shape node to use for weights splitting.
        influence_transforms (list[str]): A list of transform names that the weights need to be split along.
        degree (int, optional): Degree of the spline curve. Defaults to 2.
        vertex_indices: A list of vertex indices to output weights for.
        debug_curve: If True a curve node will be created for debug purposes.
    Returns:
        list[list[tuple[Any, float]]]: A list of weights per vertex. Each entry is a list of tuples,
        where each tuple contains a influence transform and its corresponding influence weight on the vertex.
    """
    msel: MSelectionList = MSelectionList()
    msel.add(mesh_shape)
    msel.add(surface_shape)
    mesh_dag: MDagPath = msel.getDagPath(0)
    surface_dag: MDagPath = msel.getDagPath(1)

    # make the function sets and data on the surface
    fn_mesh: om2.MFnMesh = om2.MFnMesh(mesh_dag)
    fn_surface = MFnNurbsSurface(surface_dag)
    surface_v_range: tuple[float, float] = cmds.getAttr(f"{surface_shape}.minMaxRangeV")[0]

    # get the points in world space
    mesh_points: MPointArray = get_mesh_points(fn_mesh=fn_mesh, vertex_indices=vertex_indices)

    # iterate over the points and get the closest parameter
    parameters: list[float] = []
    for i, point in enumerate(mesh_points):  # type: ignore
        parameter: float = fn_surface.closestPoint(point, space=om2.MSpace.kObject)[2]
        new_parameter = remap(
            input=parameter,
            input_range=(surface_v_range),
            output_range=(0, len(influence_transforms)),
        )
        parameters.append(new_parameter)

    spline_weights_per_vertex: list[list[tuple[CV, float]]] = spline.get_weights_along_spline(
        cvs=influence_transforms, parameters=parameters, degree=degree
    )

    return spline_weights_per_vertex
