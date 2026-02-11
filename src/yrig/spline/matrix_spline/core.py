from typing import Sequence

import maya.cmds as cmds
from maya.api.OpenMaya import (
    MDoubleArray,
    MFnNurbsCurve,
    MFnNurbsCurveData,
    MObject,
    MPoint,
    MPointArray,
    MSpace,
)

from yrig.maya_api import node
from yrig.spline.math import generate_knots, get_point_on_spline
from yrig.structs.transform import Vector3


class MatrixSpline:
    def __init__(
        self,
        cv_transforms: Sequence[str],
        degree: int = 3,
        knots: Sequence[float] | None = None,
        periodic: bool = False,
        name: str | None = None,
    ) -> None:
        """
        A matrix-based B-spline representation driven by transform nodes (CVs).

        Encapsulates a B-spline where each control vertex (CV) is represented by a transform
        in the scene. Instead of interpolating only point positions, the spline blends full
        4x4 matrices derived from each CV’s transform, with scale encoded in the matrix’s
        empty elements.

        Args:
            cv_transforms (list[str]): Transform node names used as control vertices.
            degree (int, optional): Spline degree. Defaults to 3.
            knots (list[float] | None, optional): Knot vector. If None, a suitable vector
                is generated from the CV count, degree, and periodic setting.
            periodic (bool, optional): Whether the spline is periodic (closed). Defaults to False.
            name (str | None, optional): Base name for created scene nodes. Defaults to "matrix_spline".
        """
        self.pinned_transforms: list[str] = []
        self.curve: str | None = None
        self.periodic: bool = periodic
        self.degree: int = degree
        self.cv_transforms: list[str] = list(cv_transforms)
        number_of_cvs: int = len(cv_transforms) + (periodic * degree)
        self.knots: list[float] = (
            list(knots)
            if knots is not None
            else generate_knots(count=number_of_cvs, degree=degree, periodic=periodic)
        )
        self.name: str = name if name is not None else "matrix_spline"

        cv_matrices: list[str] = []
        cv_position_attrs: list[tuple[str, str, str]] = []
        for index, cv_transform in enumerate(cv_transforms):
            # Remove scale and shear from matrix since they will interfere with the
            # linear interpolation of the basis vectors (causing flipping)
            pick_matrix = node.PickMatrixNode(name=f"{cv_transform}_pick_matrix")
            pick_matrix.input_matrix.connect_from(f"{cv_transform}.matrix")
            pick_matrix.use_shear.set(False)
            pick_matrix.use_scale.set(False)
            # Add nodes to connect individual values from the matrix,
            # I don't know why maya makes us do this instead of just connecting directly
            deconstruct_matrix_attribute = pick_matrix.output_matrix
            row1 = node.RowFromMatrixNode(name=f"{cv_transform}_row1")
            row1.matrix.connect_from(deconstruct_matrix_attribute)
            row1.input.set(0)
            row2 = node.RowFromMatrixNode(name=f"{cv_transform}_row2")
            row2.matrix.connect_from(deconstruct_matrix_attribute)
            row2.input.set(1)
            row3 = node.RowFromMatrixNode(name=f"{cv_transform}_row3")
            row3.matrix.connect_from(deconstruct_matrix_attribute)
            row3.input.set(2)
            row4 = node.RowFromMatrixNode(name=f"{cv_transform}_row4")
            row4.matrix.connect_from(deconstruct_matrix_attribute)
            row4.input.set(3)

            # Rebuild the matrix but encode the scale into the empty values in the matrix
            # (this needs to be extracted after the weighted matrix sum)
            cv_matrix = node.FourByFourMatrixNode(name=f"{cv_transform}_cv_matrix")
            row1.output.x.connect_to(cv_matrix.in_00)
            row1.output.y.connect_to(cv_matrix.in_01)
            row1.output.z.connect_to(cv_matrix.in_02)
            cmds.connectAttr(f"{cv_transform}.scaleX", str(cv_matrix.in_03))

            row2.output.x.connect_to(cv_matrix.in_10)
            row2.output.y.connect_to(cv_matrix.in_11)
            row2.output.z.connect_to(cv_matrix.in_12)
            cmds.connectAttr(f"{cv_transform}.scaleY", str(cv_matrix.in_13))

            row3.output.x.connect_to(cv_matrix.in_20)
            row3.output.y.connect_to(cv_matrix.in_21)
            row3.output.z.connect_to(cv_matrix.in_22)
            cmds.connectAttr(f"{cv_transform}.scaleZ", str(cv_matrix.in_23))

            row4.output.x.connect_to(cv_matrix.in_30)
            row4.output.y.connect_to(cv_matrix.in_31)
            row4.output.z.connect_to(cv_matrix.in_32)
            row4.output.w.connect_to(cv_matrix.in_33)

            cv_matrices.append(str(cv_matrix.output))
            cv_position_attrs.append((str(row4.output.x), str(row4.output.y), str(row4.output.z)))

        # If the curve is periodic there are we need to re-add CVs that move together.
        if periodic:
            for i in range(degree):
                cv_matrices.append(cv_matrices[i])

        self.cv_matrices: list[str] = cv_matrices
        self.cv_position_attrs: list[tuple[str, str, str]] = cv_position_attrs

    def create_bound_curve(self, curve_name: str | None = None, curve_parent: str | None = None):
        """
        Creates a NURBS curve driven by the MatrixSpline’s control transforms.

        This function builds a NURBS curve whose control points are bound directly to
        the CV transforms of the MatrixSpline. This is useful for having calculating a
        live attribute for the MatrixSpline arc length for example.

        Args:
            matrix_spline (MatrixSpline): The MatrixSpline instance providing CVs, knots,
                and degree information.
            curve_name (str | None, optional): Optional name for the curve transform.
            curve_parent (str | None, optional): Optional parent transform to parent the
                created curve under.

        Returns:
            str: The name of the created curve transform node.
        """
        return bound_curve_from_matrix_spline(
            matrix_spline=self, curve_name=curve_name, curve_parent=curve_parent
        )

    def closest_parameter(self, position: MPoint | tuple[float, float, float]) -> float:
        """
        Finds the closest parameter value on the MatrixSpline to a given 3D position in world space.

        Args:
            position: The world-space point to project onto the spline.

        Returns:
            float: The curve parameter value (in knot space) at the closest point to the input position.
        """
        return closest_parameter_on_matrix_spline(matrix_spline=self, position=position)

    def get_point(self, parameter: float, normalize_parameter: bool = True) -> MPoint:
        return get_point_on_matrix_spline(matrix_spline=self, parameter=parameter)


def get_matrix_spline_mfn_curve(matrix_spline: MatrixSpline) -> tuple[MFnNurbsCurve, MObject]:
    cv_transforms: list[str] = matrix_spline.cv_transforms
    cv_positions: MPointArray = MPointArray()
    for transform in cv_transforms:
        cv_position: tuple[float, float, float] = cmds.xform(  # type: ignore
            transform, query=True, worldSpace=True, translation=True
        )
        cv_positions.append(MPoint(*cv_position))
    maya_knots: list[float] = matrix_spline.knots[1:-1]

    fn_data: MFnNurbsCurveData = MFnNurbsCurveData()
    data_obj: MObject = fn_data.create()
    fn_curve: MFnNurbsCurve = MFnNurbsCurve()
    fn_curve.create(
        cv_positions,
        MDoubleArray(maya_knots),
        matrix_spline.degree,
        MFnNurbsCurve.kOpen if not matrix_spline.periodic else MFnNurbsCurve.kPeriodic,
        False,  # create2D
        False,  # not rational
        data_obj,
    )
    return fn_curve, data_obj


def closest_parameter_on_matrix_spline(
    matrix_spline: MatrixSpline, position: MPoint | tuple[float, float, float]
) -> float:
    """
    Finds the closest parameter value on a spline (defined by a MatrixSpline) to a given 3D position.

    Args:
        matrix_spline: Spline definition object.
        position: The world-space point to project onto the spline.

    Returns:
        float: The curve parameter value (in knot space) at the closest point to the input position.
    """
    fn_curve, data_obj = get_matrix_spline_mfn_curve(matrix_spline)
    test_point = position if isinstance(position, MPoint) else MPoint(*position)
    parameter: float = fn_curve.closestPoint(test_point, space=MSpace.kObject)[1]

    return parameter


def get_point_on_matrix_spline(
    matrix_spline: MatrixSpline, parameter: float, normalize_paramter: bool = True
) -> MPoint:
    cv_transforms: list[str] = matrix_spline.cv_transforms
    cv_positions: list[Vector3] = []
    for transform in cv_transforms:
        cv_position: tuple[float, float, float] = cmds.xform(  # type: ignore
            transform, query=True, worldSpace=True, translation=True
        )
        cv_positions.append(Vector3(*cv_position))

    position = get_point_on_spline(
        cv_positions=cv_positions,
        t=parameter,
        degree=matrix_spline.degree,
        knots=matrix_spline.knots,
        normalize_parameter=normalize_paramter,
    )
    point: MPoint = MPoint(position.x, position.y, position.z)
    return point


def bound_curve_from_matrix_spline(
    matrix_spline: MatrixSpline, curve_name: str | None = None, curve_parent: str | None = None
) -> str:
    """
    Creates a NURBS curve driven by a MatrixSpline’s control transforms.

    This function builds a NURBS curve whose control points are bound directly to
    the CV transforms of a given MatrixSpline. This is useful for having calculating a
    live attribute for the MatrixSpline arc length for example.

    Args:
        matrix_spline (MatrixSpline): The MatrixSpline instance providing CVs, knots,
            and degree information.
        curve_name (str | None, optional): Optional name for the curve transform.
        curve_parent (str | None, optional): Optional parent transform to parent the
            created curve under.

    Returns:
        str: The name of the created curve transform node.
    """
    curve_transform_name = curve_name if curve_name is not None else f"{matrix_spline.name}_curve"
    maya_knots: Sequence[float] = matrix_spline.knots[1:-1]
    extended_cvs: Sequence[str] = (
        (matrix_spline.cv_transforms + matrix_spline.cv_transforms[: matrix_spline.degree])
        if matrix_spline.periodic
        else matrix_spline.cv_transforms
    )
    curve_transform: str = cmds.curve(
        name=curve_transform_name,
        point=[  # type: ignore
            cmds.xform(cv, query=True, worldSpace=True, translation=True) for cv in extended_cvs
        ],
        periodic=matrix_spline.periodic,
        knot=maya_knots,
        degree=matrix_spline.degree,
    )
    if curve_parent is not None:
        cmds.parent(curve_transform, curve_parent, relative=True)

    for index, cv_position_attrs in enumerate(matrix_spline.cv_position_attrs):
        cmds.connectAttr(cv_position_attrs[0], f"{curve_transform}.controlPoints[{index}].xValue")
        cmds.connectAttr(cv_position_attrs[1], f"{curve_transform}.controlPoints[{index}].yValue")
        cmds.connectAttr(cv_position_attrs[2], f"{curve_transform}.controlPoints[{index}].zValue")
    return curve_transform
