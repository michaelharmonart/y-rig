import maya.cmds as cmds
from maya.api.OpenMaya import (
    MDoubleArray,
    MFnNurbsCurve,
    MPointArray,
    MSelectionList,
    MSpace,
)

from yrig.structs.transform import Vector3


def maya_to_standard_knots(
    knots: list[float], degree: int = 3, periodic: bool = False
) -> list[float]:
    # Refer to https://openusd.org/dev/api/class_usd_geom_nurbs_curves.html#details
    # The above only works with uniform knots, so this is generalized to higher order and non-uniform knots
    # based on info found here https://developer.rhino3d.com/guides/opennurbs/periodic-curves-and-surfaces/
    # Although there is a typo in the above doc, k[(cv_count)+i] should be k[(cv_count - 1)+i]
    # Don't ask how long it took me to find that out
    """
    Convert Maya-style knot vector to a 'standard' knot vector.

    Args:
        knots: The Maya-style knot sequence (missing the first and last values).
        degree: Degree of the NURBS curve.  Defaults to ``3`` (cubic).
        periodic: Whether the curve is periodic (closed with ``C^(degree-1)``
            continuity).  Defaults to ``False``.

    Returns:
        A new knot vector with the two missing boundary values restored,
        suitable for use with standard B-spline / NURBS evaluation.
    """

    new_knots: list[float] = knots.copy()

    # add placeholders for first/last values
    new_knots.insert(0, 0.0)
    new_knots.append(0.0)

    # A cubic periodic knot vector looks like: [a,b,c,d,e, ...,  p+a,p+b,p+c,p+d,p+e]
    # offset is the length of the repeated indices, in the above case it would be 5
    # (degree is multiplied by 2 since degree is both part of the iterator and the indexing equation):
    # -degree < i < degree (max i here is used to calculate p)
    # k[(degree-1)+i+1] - k[(degree-1)+i] = k[(cv_count-1)+i+1] - k[(cv_count-1)+i]
    # (degree-1)+i and (cv_count-1)+i can now both be substituted with the offset

    offset: int = (degree * 2) - 1

    if periodic:
        new_knots[0] = new_knots[1] - (new_knots[-(offset - 1)] - new_knots[-offset])
        new_knots[-1] = new_knots[-2] + (new_knots[offset] - new_knots[offset - 1])
    else:
        new_knots[0] = new_knots[1]
        new_knots[-1] = new_knots[-2]
    return new_knots


def get_knots(curve_shape: str) -> list[float]:
    """Retrieve the standard (full) knot vector for a NURBS curve shape.

    Reads the internal Maya knot vector via the API and converts it to the
    standard form by restoring the two omitted boundary knot values.  The
    returned vector has length ``num_cvs + degree + 1`` and is suitable for
    direct use with the spline evaluation utilities in `spline.math`.

    Args:
        curve_shape: The name of a NURBS curve **shape** node (not the
            transform).

    Returns:
        The full knot vector as a list of floats.

    See Also:
        `maya_to_standard_knots` for details on the conversion.
    """

    sel: MSelectionList = MSelectionList()
    sel.add(curve_shape)
    curve_obj = sel.getDependNode(0)
    fn_curve: MFnNurbsCurve = MFnNurbsCurve(curve_obj)

    knots_array: MDoubleArray = fn_curve.knots()
    knots: list[float] = [knot for knot in knots_array]

    # Now convert the knot vector from the Maya form to the standard form by filling out the missing values.
    degree: int = cmds.getAttr(f"{curve_shape}.degree")
    periodic: bool = cmds.getAttr(f"{curve_shape}.form") == 2
    out_knots: list[float] = maya_to_standard_knots(knots=knots, degree=degree, periodic=periodic)
    return out_knots


def get_cvs(curve_shape: str) -> list[Vector3]:
    """Retrieve the world-space positions of all CVs on a NURBS curve shape.

    Uses the Maya API (``MFnNurbsCurve.cvPositions``) to efficiently query
    every control vertex and returns them as lightweight
    `Vector3` instances.

    Args:
        curve_shape: The name of a NURBS curve **shape** node (not the
            transform).

    Returns:
        An ordered list of `Vector3` objects,
        one per CV, in world space.
    """
    sel: MSelectionList = MSelectionList()
    sel.add(curve_shape)
    dag_path = sel.getDagPath(0)
    fn_curve: MFnNurbsCurve = MFnNurbsCurve(dag_path)

    cv_positions: MPointArray = fn_curve.cvPositions(space=MSpace.kWorld)
    positions: list[Vector3] = [Vector3(point.x, point.y, point.z) for point in cv_positions]
    return positions


def get_cv_weights(curve_shape: str) -> list[float]:
    """
    Gets the weights of all CVs for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of CV weight values.
    """
    sel: MSelectionList = MSelectionList()
    sel.add(curve_shape)
    dag_path = sel.getDagPath(0)
    fn_curve: MFnNurbsCurve = MFnNurbsCurve(dag_path)

    cv_positions: MPointArray = fn_curve.cvPositions(space=MSpace.kWorld)
    weights: list[float] = [point.w for point in cv_positions]
    return weights
