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
from yrig.maya_api.attribute import (
    MatrixAttribute,
    ScalarAttribute,
    Vector4Attribute,
)
from yrig.spline.math import (
    generate_knots,
    point_on_spline_weights,
    tangent_on_spline_weights,
)
from yrig.structs.transform import Vector3


class MatrixSpline:
    def __init__(
        self,
        cv_transforms: list[str],
        degree: int = 3,
        knots: list[float] | None = None,
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
            name (str | None, optional): Base name for created scene nodes. Defaults to "MatrixSpline".
        """
        self.pinned_transforms: list[str] = []
        self.pinned_drivers: list[str] = []
        self.curve: str | None = None
        self.periodic: bool = periodic
        self.degree: int = degree
        self.cv_transforms: list[str] = cv_transforms
        number_of_cvs: int = len(cv_transforms) + (periodic * degree)
        self.knots: list[float] = (
            knots
            if knots is not None
            else generate_knots(count=number_of_cvs, degree=degree, periodic=periodic)
        )
        self.name: str = name if name is not None else "MatrixSpline"

        cv_matrices: list[str] = []
        cv_position_attrs: list[tuple[str, str, str]] = []
        for index, cv_transform in enumerate(cv_transforms):
            # Remove scale and shear from matrix since they will interfere with the
            # linear interpolation of the basis vectors (causing flipping)
            pick_matrix = node.PickMatrixNode(name=f"{cv_transform}_PickMatrix")
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
            cv_matrix = node.FourByFourMatrixNode(name=f"{cv_transform}_CvMatrix")
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
            cv_position_attrs.append((str({row4.output.x}), str(row4.output.y), str(row4.output.z)))

        # If the curve is periodic there are we need to re-add CVs that move together.
        if periodic:
            for i in range(degree):
                cv_matrices.append(cv_matrices[i])

        self.cv_matrices: list[str] = cv_matrices
        self.cv_position_attrs: list[tuple[str, str, str]] = cv_position_attrs


def bound_curve_from_matrix_spline(
    matrix_spline: MatrixSpline, curve_parent: str | None = None
) -> str:
    """
    Creates a NURBS curve driven by a MatrixSpline’s control transforms.

    This function builds a NURBS curve whose control points are bound directly to
    the CV transforms of a given MatrixSpline. This is useful for having calculating a
    live attribute for the MatrixSpline arc length for example.

    Args:
        matrix_spline (MatrixSpline): The MatrixSpline instance providing CVs, knots,
            and degree information.
        curve_parent (str | None, optional): Optional parent transform to parent the
            created curve under. If provided, the curve is parented relatively.

    Returns:
        str: The name of the created curve transform node.
    """
    maya_knots: list[float] = matrix_spline.knots[1:-1]
    extended_cvs: list[str] = (
        (matrix_spline.cv_transforms + matrix_spline.cv_transforms[: matrix_spline.degree])
        if matrix_spline.periodic
        else matrix_spline.cv_transforms
    )
    curve_transform: str = cmds.curve(
        name=f"{matrix_spline.name}_Curve",
        point=[
            cmds.xform(cv, query=True, worldSpace=True, translation=True)
            for cv in extended_cvs  # type: ignore
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


def closest_point_on_matrix_spline(
    matrix_spline: MatrixSpline, position: tuple[float, float, float]
) -> float:
    """
    Finds the closest parameter value on a spline (defined by a MatrixSpline) to a given 3D position.

    Args:
        matrix_spline: Spline definition object.
        position: The world-space point to project onto the spline.

    Returns:
        float: The curve parameter value (in knot space) at the closest point to the input position.
    """
    knots: list[float] = matrix_spline.knots
    degree: int = matrix_spline.degree
    periodic: bool = matrix_spline.periodic
    cv_transforms: list[str] = matrix_spline.cv_transforms
    cv_positions: MPointArray = MPointArray()
    for transform in cv_transforms:
        cv_position: tuple[float, float, float] = cmds.xform(  # type: ignore
            transform, query=True, worldSpace=True, translation=True
        )
        cv_positions.append(MPoint(*cv_position))
    maya_knots: list[float] = knots[1:-1]

    fn_data: MFnNurbsCurveData = MFnNurbsCurveData()
    data_obj: MObject = fn_data.create()
    fn_curve: MFnNurbsCurve = MFnNurbsCurve()
    fn_curve.create(
        cv_positions,
        MDoubleArray(maya_knots),
        degree,
        MFnNurbsCurve.kOpen if not periodic else MFnNurbsCurve.kPeriodic,
        False,  # create2D
        False,  # not rational
        data_obj,
    )

    parameter: float = fn_curve.closestPoint(
        MPoint(position[0], position[1], position[2]), space=MSpace.kObject
    )[1]

    return parameter


def pin_to_matrix_spline(
    matrix_spline: MatrixSpline,
    pinned_transform: str,
    parameter: float,
    normalize_parameter: bool = True,
    stretch: bool = True,
    primary_axis: tuple[int, int, int] | None = (0, 1, 0),
    secondary_axis: tuple[int, int, int] | None = (0, 0, 1),
    twist: bool = True,
    align_tangent: bool = True,
) -> None:
    """
    Pins a transform to a matrix spline at a given parameter along the curve.

    Args:
        matrix_spline: The matrix spline data object.
        pinned_transform: Transform to pin to the spline.
        parameter: Position along the spline (0–1).
        stretch: Whether to apply automatic scaling along the spline tangent.
        primary_axis (tuple[int, int, int], optional): Local axis of the pinned
            transform that should aim down the spline tangent. Must be one of
            the cardinal axes (±X, ±Y, ±Z). Defaults to (0, 1, 0) (the +Y axis).
        secondary_axis (tuple[int, int, int], optional): Local axis of the pinned
            transform that should be aligned to a secondary reference direction
            from the spline. Used to resolve orientation. Must be one of the
            cardinal axes (±X, ±Y, ±Z) and orthogonal to ``primary_axis``.
            Defaults to (0, 0, 1) (the +Z axis).
        twist (bool): When True the twist is calculated by averaging the secondary axis vector
            as the up vector for the aim matrix. If False no vector is set and the orientation is the swing
            part of a swing twist decomposition.
        align_tangent: When True the pinned segments will align their primary axis along the spline.
    Returns:
        None
    """
    if not primary_axis:
        primary_axis = (0, 1, 0)
    if not secondary_axis:
        secondary_axis = (0, 0, 1)

    CARDINALS = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    if tuple(primary_axis) not in CARDINALS or tuple(secondary_axis) not in CARDINALS:
        raise ValueError(
            "primary_axis and secondary_axis must be one of the cardinal axes (±X, ±Y, ±Z)."
        )

    cv_matrices: list[str] = matrix_spline.cv_matrices
    degree: int = matrix_spline.degree
    knots: list[float] = matrix_spline.knots
    segment_name: str = pinned_transform

    # Create node that blends the matrices based on the calculated DeBoor weights.
    blended_matrix = node.WtAddMatrixNode(name=f"{segment_name}_BaseMatrix")
    point_weights = point_on_spline_weights(
        cvs=cv_matrices, t=parameter, degree=degree, knots=knots, normalize=normalize_parameter
    )
    for index, point_weight in enumerate(point_weights):
        blended_matrix.weight_matrix[index].weight_in.set(point_weight[1])
        blended_matrix.weight_matrix[index].matrix_in.connect_from(point_weight[0])

    # Create nodes to access the values of the blended matrix node.
    deconstruct_matrix_attribute = blended_matrix.matrix_sum
    blended_matrix_row1 = node.RowFromMatrixNode(name=f"{blended_matrix}_row1")
    blended_matrix_row1.input.set(0)
    blended_matrix_row1.matrix.connect_from(deconstruct_matrix_attribute)

    blended_matrix_row2 = node.RowFromMatrixNode(name=f"{blended_matrix}_row2")
    blended_matrix_row2.input.set(1)
    blended_matrix_row2.matrix.connect_from(deconstruct_matrix_attribute)

    blended_matrix_row3 = node.RowFromMatrixNode(name=f"{blended_matrix}_row3")
    blended_matrix_row3.input.set(2)
    blended_matrix_row3.matrix.connect_from(deconstruct_matrix_attribute)

    blended_matrix_row4 = node.RowFromMatrixNode(name=f"{blended_matrix}_row4")
    blended_matrix_row4.input.set(3)
    blended_matrix_row4.matrix.connect_from(deconstruct_matrix_attribute)

    axis_to_row: dict[tuple[int, int, int], node.RowFromMatrixNode] = {
        (1, 0, 0): blended_matrix_row1,
        (0, 1, 0): blended_matrix_row2,
        (0, 0, 1): blended_matrix_row3,
        (-1, 0, 0): blended_matrix_row1,  # flipped
        (0, -1, 0): blended_matrix_row2,
        (0, 0, -1): blended_matrix_row3,
    }

    tangent_vector_node: node.MultiplyPointByMatrixNode | None = None
    rigid_matrix_output: MatrixAttribute
    if align_tangent:
        blended_tangent_matrix = node.WtAddMatrixNode(name=f"{segment_name}_TangentMatrix")
        tangent_weights = tangent_on_spline_weights(
            cvs=cv_matrices, t=parameter, degree=degree, knots=knots, normalize=normalize_parameter
        )
        for index, tangent_weight in enumerate(tangent_weights):
            blended_tangent_matrix.weight_matrix[index].weight_in.set(tangent_weight[1])
            blended_tangent_matrix.weight_matrix[index].matrix_in.connect_from(tangent_weight[0])

        tangent_vector_node = node.MultiplyPointByMatrixNode(
            name=f"{blended_tangent_matrix}_TangentVector"
        )
        blended_tangent_matrix.matrix_sum.connect_to(tangent_vector_node.input_matrix)

        # Create aim matrix node.
        aim_matrix = node.AimMatrixNode(name=f"{segment_name}_AimMatrix")
        aim_matrix.primary.mode.set(2)
        aim_matrix.primary.input_axis.set(primary_axis)
        tangent_vector_node.output.connect_to(aim_matrix.primary.target_vector)

        secondary_row: node.RowFromMatrixNode | None = axis_to_row.get(secondary_axis)
        if secondary_row and twist:
            aim_matrix.secondary.mode.set(2)
            aim_matrix.secondary.input_axis.set(secondary_axis)
            secondary_row.output.x.connect_to(aim_matrix.secondary.target_vector.x)
            secondary_row.output.y.connect_to(aim_matrix.secondary.target_vector.y)
            secondary_row.output.z.connect_to(aim_matrix.secondary.target_vector.z)
        else:
            aim_matrix.secondary.mode.set(0)
        rigid_matrix = aim_matrix
        rigid_matrix_output = aim_matrix.output_matrix
    else:
        pick_matrix = node.PickMatrixNode(name=f"{segment_name}_Ortho")
        pick_matrix.use_translate.set(True)
        pick_matrix.use_rotate.set(True)
        pick_matrix.use_scale.set(False)
        pick_matrix.use_shear.set(False)
        deconstruct_matrix_attribute.connect_to(pick_matrix.input_matrix)
        rigid_matrix = pick_matrix
        rigid_matrix_output = pick_matrix.output_matrix

    # Create nodes to access the values of the rigid matrix (aim matrix or pick matrix) node.
    rigid_matrix_row1 = node.RowFromMatrixNode(name=f"{rigid_matrix}_row1")
    rigid_matrix_row1.matrix.connect_from(rigid_matrix_output)
    rigid_matrix_row1.input.set(0)

    rigid_matrix_row2 = node.RowFromMatrixNode(name=f"{rigid_matrix}_row2")
    rigid_matrix_row2.matrix.connect_from(rigid_matrix_output)
    rigid_matrix_row2.input.set(1)

    rigid_matrix_row3 = node.RowFromMatrixNode(name=f"{rigid_matrix}_row3")
    rigid_matrix_row3.matrix.connect_from(rigid_matrix_output)
    rigid_matrix_row3.input.set(2)

    tangent_scale_attr: ScalarAttribute | None = None
    if align_tangent and stretch and tangent_vector_node is not None:
        # Get tangent vector magnitude
        tangent_vector_length = node.LengthNode(name=f"{segment_name}_tangentVectorLength")
        tangent_vector_node.output.connect_to(tangent_vector_length.input)
        tangent_vector_length_scaled: node.MultiplyNode = node.MultiplyNode(
            name=f"{segment_name}_tangentVectorLengthScaled"
        )
        tangent_vector_length.output.connect_to(tangent_vector_length_scaled.input[0])
        tangent_sample = tangent_vector_node.output.get()
        tangent_length = Vector3(tangent_sample[0], tangent_sample[1], tangent_sample[2]).length()
        if tangent_length == 0:
            raise RuntimeError(
                f"{pinned_transform} had a tangent magnitude of 0 and wasn't able to be pinned with stretching enabled."
            )
        tangent_vector_length_scaled.input[1].set(1 / tangent_length)
        tangent_scale_attr = tangent_vector_length_scaled.output

    def is_same_axis(axis1: tuple[int, int, int], axis2: tuple[int, int, int]) -> bool:
        # Compare absolute values to handle flips: (0,1,0) == (0,-1,0)
        return tuple(abs(v) for v in axis1) == tuple(abs(v) for v in axis2)

    def scale_vector(
        vector_attr: Vector4Attribute,
        scale_attr: ScalarAttribute,
        node_name: str,
        axis: tuple[int, int, int],
    ) -> node.MultiplyDivideNode:
        scale_node = node.MultiplyDivideNode(name=node_name)
        scale_node.input1.x.connect_from(vector_attr.x)
        scale_node.input1.y.connect_from(vector_attr.y)
        scale_node.input1.z.connect_from(vector_attr.z)

        scalar_to_connect: ScalarAttribute
        if stretch and tangent_scale_attr is not None and is_same_axis(axis, primary_axis):
            scalar_to_connect = tangent_scale_attr
        else:
            scalar_to_connect = scale_attr

        scale_node.input2.x.connect_from(scalar_to_connect)
        scale_node.input2.y.connect_from(scalar_to_connect)
        scale_node.input2.z.connect_from(scalar_to_connect)

        return scale_node

    # Create Nodes to re-apply scale
    X_AXIS = (1, 0, 0)
    Y_AXIS = (0, 1, 0)
    Z_AXIS = (0, 0, 1)

    x_scaled = scale_vector(
        node_name=f"{segment_name}_xScale",
        vector_attr=rigid_matrix_row1.output,
        scale_attr=blended_matrix_row1.output.w,
        axis=X_AXIS,
    )
    y_scaled = scale_vector(
        node_name=f"{segment_name}_yScale",
        vector_attr=rigid_matrix_row2.output,
        scale_attr=blended_matrix_row2.output.w,
        axis=Y_AXIS,
    )
    z_scaled = scale_vector(
        node_name=f"{segment_name}_zScale",
        vector_attr=rigid_matrix_row3.output,
        scale_attr=blended_matrix_row3.output.w,
        axis=Z_AXIS,
    )

    # Rebuild the matrix
    output_matrix = node.FourByFourMatrixNode(name=f"{segment_name}_OutputMatrix")
    x_scaled.output.x.connect_to(output_matrix.in_00)
    x_scaled.output.y.connect_to(output_matrix.in_01)
    x_scaled.output.z.connect_to(output_matrix.in_02)

    y_scaled.output.x.connect_to(output_matrix.in_10)
    y_scaled.output.y.connect_to(output_matrix.in_11)
    y_scaled.output.z.connect_to(output_matrix.in_12)

    z_scaled.output.x.connect_to(output_matrix.in_30)
    z_scaled.output.y.connect_to(output_matrix.in_31)
    z_scaled.output.z.connect_to(output_matrix.in_32)

    output_matrix.output.connect_to(f"{pinned_transform}.offsetParentMatrix")
    matrix_spline.pinned_transforms.append(pinned_transform)
