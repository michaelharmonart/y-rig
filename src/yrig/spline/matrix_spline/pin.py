from typing import Sequence

import maya.cmds as cmds

from yrig.maya_api import node
from yrig.maya_api.attribute import MatrixAttribute, ScalarAttribute, Vector4Attribute
from yrig.spline.math import point_on_spline_weights, resample, tangent_on_spline_weights
from yrig.spline.matrix_spline.core import MatrixSpline
from yrig.structs.transform import Vector3


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
    blended_matrix = node.WtAddMatrixNode(name=f"{segment_name}_base_matrix")
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
        blended_tangent_matrix = node.WtAddMatrixNode(name=f"{segment_name}_tangent_matrix")
        tangent_weights = tangent_on_spline_weights(
            cvs=cv_matrices, t=parameter, degree=degree, knots=knots, normalize=normalize_parameter
        )
        for index, tangent_weight in enumerate(tangent_weights):
            blended_tangent_matrix.weight_matrix[index].weight_in.set(tangent_weight[1])
            blended_tangent_matrix.weight_matrix[index].matrix_in.connect_from(tangent_weight[0])

        tangent_vector_node = node.MultiplyPointByMatrixNode(
            name=f"{blended_tangent_matrix}_tangent_vector"
        )
        blended_tangent_matrix.matrix_sum.connect_to(tangent_vector_node.input_matrix)

        # Create aim matrix node.
        aim_matrix = node.AimMatrixNode(name=f"{segment_name}_aim_matrix")
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
        pick_matrix = node.PickMatrixNode(name=f"{segment_name}_ortho")
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
        tangent_vector_length = node.LengthNode(name=f"{segment_name}_tangent_vector_length")
        tangent_vector_node.output.connect_to(tangent_vector_length.input)
        tangent_vector_length_scaled: node.MultiplyNode = node.MultiplyNode(
            name=f"{segment_name}_tangent_vector_length_scaled"
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
        node_name=f"{segment_name}_x_scale",
        vector_attr=rigid_matrix_row1.output,
        scale_attr=blended_matrix_row1.output.w,
        axis=X_AXIS,
    )
    y_scaled = scale_vector(
        node_name=f"{segment_name}_y_scale",
        vector_attr=rigid_matrix_row2.output,
        scale_attr=blended_matrix_row2.output.w,
        axis=Y_AXIS,
    )
    z_scaled = scale_vector(
        node_name=f"{segment_name}_z_scale",
        vector_attr=rigid_matrix_row3.output,
        scale_attr=blended_matrix_row3.output.w,
        axis=Z_AXIS,
    )

    # Rebuild the matrix
    output_matrix = node.FourByFourMatrixNode(name=f"{segment_name}_output_matrix")
    x_scaled.output.x.connect_to(output_matrix.in_00)
    x_scaled.output.y.connect_to(output_matrix.in_01)
    x_scaled.output.z.connect_to(output_matrix.in_02)

    y_scaled.output.x.connect_to(output_matrix.in_10)
    y_scaled.output.y.connect_to(output_matrix.in_11)
    y_scaled.output.z.connect_to(output_matrix.in_12)

    z_scaled.output.x.connect_to(output_matrix.in_20)
    z_scaled.output.y.connect_to(output_matrix.in_21)
    z_scaled.output.z.connect_to(output_matrix.in_22)

    blended_matrix_row4.output.x.connect_to(output_matrix.in_30)
    blended_matrix_row4.output.y.connect_to(output_matrix.in_31)
    blended_matrix_row4.output.z.connect_to(output_matrix.in_32)

    output_matrix.output.connect_to(f"{pinned_transform}.offsetParentMatrix")
    matrix_spline.pinned_transforms.append(pinned_transform)


def pin_transforms_to_matrix_spline(
    matrix_spline: MatrixSpline,
    pinned_transforms: Sequence[str],
    padded: bool = True,
    stretch: bool = True,
    arc_length: bool = True,
    primary_axis: tuple[int, int, int] | None = (0, 1, 0),
    secondary_axis: tuple[int, int, int] | None = (0, 0, 1),
    twist: bool = True,
    align_tangent: bool = True,
) -> MatrixSpline:
    """
    Takes a set of transforms (cvs) and creates a matrix spline with controls and deformation joints.
    Args:
        matrix_spline: The matrix spline defention that will drive the pinned transforms.
        pinned_transforms: These transforms will be constrained to the spline.
        padded: When True, segments are sampled such that the end points have half a segment of spacing from the ends of the spline.
        stretch: Whether to apply automatic scaling along the spline tangent.
        arc_length: When True, the parameters for the spline will be even according to arc length.
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
        matrix_spline: The matrix spline.
    """
    segments = len(pinned_transforms)
    cv_positions: list[Vector3] = []

    for transform in pinned_transforms:
        position: tuple[float, float, float] = cmds.xform(  # type: ignore
            transform, query=True, worldSpace=True, translation=True
        )
        cv_positions.append(Vector3(*position))

    segment_parameters: list[float] = resample(
        cv_positions=cv_positions,
        number_of_points=segments,
        degree=matrix_spline.degree,
        knots=matrix_spline.knots,
        periodic=matrix_spline.periodic,
        padded=padded,
        arc_length=arc_length,
        normalize_parameter=False,
    )

    for transform, parameter in zip(pinned_transforms, segment_parameters):
        pin_to_matrix_spline(
            matrix_spline=matrix_spline,
            pinned_transform=transform,
            parameter=parameter,
            stretch=stretch,
            primary_axis=primary_axis,
            secondary_axis=secondary_axis,
            normalize_parameter=False,
            twist=twist,
            align_tangent=align_tangent,
        )
    return matrix_spline
