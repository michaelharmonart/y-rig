from collections.abc import Sequence

from maya import cmds

from yrig.maya_api import node
from yrig.maya_api.attribute import (
    MatrixAttribute,
    ScalarAttribute,
    Vector3Attribute,
    Vector4Attribute,
)
from yrig.maya_api.node import AimMatrixNode, PickMatrixNode
from yrig.spline.math import point_on_spline_weights, resample, tangent_on_spline_weights
from yrig.spline.matrix_spline.core import MatrixSpline
from yrig.structs.transform import Vector3
from yrig.transform.utils import zero_transform

CARDINALS = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
X_AXIS = (1, 0, 0)
Y_AXIS = (0, 1, 0)
Z_AXIS = (0, 0, 1)


def _is_same_axis(axis1: tuple[int, int, int], axis2: tuple[int, int, int]) -> bool:
    # Compare absolute values to handle flips: (0,1,0) == (0,-1,0)
    return tuple(abs(v) for v in axis1) == tuple(abs(v) for v in axis2)


def _scale_vector(
    vector_attr: Vector4Attribute,
    scale_attr: ScalarAttribute,
    node_name: str,
    axis: tuple[int, int, int],
    stretch: bool,
    tangent_scale_attr: ScalarAttribute | None,
    primary_axis: tuple[int, int, int],
    interpolate_scale: bool,
) -> Vector4Attribute | Vector3Attribute:
    create_mult: bool = False

    scalar_to_connect: ScalarAttribute
    if stretch and tangent_scale_attr is not None and _is_same_axis(axis, primary_axis):
        scalar_to_connect = tangent_scale_attr
        create_mult = True
    elif interpolate_scale:
        scalar_to_connect = scale_attr
        create_mult = True

    if create_mult:
        scale_node = node.MultiplyDivideNode.create(name=node_name)
        scale_node.input1.x.connect_from(vector_attr.x)
        scale_node.input1.y.connect_from(vector_attr.y)
        scale_node.input1.z.connect_from(vector_attr.z)
        scale_node.input2.x.connect_from(scalar_to_connect)
        scale_node.input2.y.connect_from(scalar_to_connect)
        scale_node.input2.z.connect_from(scalar_to_connect)
        return scale_node.output

    return vector_attr


def _create_tangent_scale(segment_name: str, tangent_vector: Vector3Attribute) -> ScalarAttribute:
    # Get tangent vector magnitude
    tangent_vector_length = node.LengthNode.create(name=f"{segment_name}_tangent_vector_length")
    tangent_vector.connect_to(tangent_vector_length.input)
    tangent_vector_length_scaled: node.MultiplyNode = node.MultiplyNode.create(
        name=f"{segment_name}_tangent_vector_length_scaled"
    )
    tangent_vector_length.output.connect_to(tangent_vector_length_scaled.input[0])
    tangent_sample = tangent_vector.get()
    tangent_length = Vector3(tangent_sample[0], tangent_sample[1], tangent_sample[2]).length()
    if tangent_length == 0:
        raise RuntimeError(
            f"{segment_name} had a tangent magnitude of 0 and wasn't able to be pinned with stretching enabled."
        )
    tangent_vector_length_scaled.input[1].set(1 / tangent_length)
    return tangent_vector_length_scaled.output


def _create_align_tangent(
    segment_name: str,
    cv_matrices: list[str],
    parameter: float,
    degree: int,
    knots: Sequence[float],
    normalize_parameter: bool,
    primary_axis: tuple[int, int, int],
    secondary_axis: tuple[int, int, int],
    twist: bool,
    axis_to_row: dict[tuple[int, int, int], node.RowFromMatrixNode],
) -> tuple[AimMatrixNode, MatrixAttribute, Vector3Attribute]:
    blended_tangent_matrix = node.WtAddMatrixNode.create(name=f"{segment_name}_tangent_matrix")
    tangent_weights = tangent_on_spline_weights(
        cvs=cv_matrices, t=parameter, degree=degree, knots=knots, normalize=normalize_parameter
    )
    for index, tangent_weight in enumerate(tangent_weights):
        blended_tangent_matrix.weight_matrix[index].weight_in.set(tangent_weight[1])
        blended_tangent_matrix.weight_matrix[index].matrix_in.connect_from(tangent_weight[0])

    tangent_vector_node = node.MultiplyPointByMatrixNode.create(
        name=f"{blended_tangent_matrix}_tangent_vector"
    )
    blended_tangent_matrix.matrix_sum.connect_to(tangent_vector_node.input_matrix)

    # Create aim matrix node.
    aim_matrix = node.AimMatrixNode.create(name=f"{segment_name}_aim_matrix")
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
    return aim_matrix, aim_matrix.output_matrix, tangent_vector_node.output


def _create_pick_matrix(
    segment_name: str, input_matrix: MatrixAttribute, interpolate_rotation: bool
) -> PickMatrixNode:
    pick_matrix = node.PickMatrixNode.create(name=f"{segment_name}_ortho")
    pick_matrix.use_translate.set(True)
    pick_matrix.use_rotate.set(interpolate_rotation)
    pick_matrix.use_scale.set(False)
    pick_matrix.use_shear.set(False)
    input_matrix.connect_to(pick_matrix.input_matrix)
    return pick_matrix


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
    reset_transforms: bool = True,
    interpolate_rotation: bool = True,
    interpolate_scale: bool = True,
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
        reset_transforms: When True the translate rotation scale and shear of the pinned transform will be reset
            such that the offset parent matrix can be used to drive the transform without side effects.
        interpolate_rotation: When True the rotation of the pinned transform will be interpolated with the CVs rotations.
        interpolate_scale: When True the scale of the pinned transform will be a spline interpolation of the CVs scales.

    Returns:
        None
    """
    if not primary_axis:
        primary_axis = (0, 1, 0)
    if not secondary_axis:
        secondary_axis = (0, 0, 1)

    if tuple(primary_axis) not in CARDINALS or tuple(secondary_axis) not in CARDINALS:
        raise ValueError(
            "primary_axis and secondary_axis must be one of the cardinal axes (±X, ±Y, ±Z)."
        )

    cv_matrices: list[str] = matrix_spline.cv_matrices
    degree: int = matrix_spline.degree
    knots: list[float] = matrix_spline.knots
    segment_name: str = pinned_transform

    # Create node that blends the matrices based on the calculated DeBoor weights.
    blended_matrix = node.WtAddMatrixNode.create(name=f"{segment_name}_base_matrix")
    point_weights = point_on_spline_weights(
        cvs=cv_matrices, t=parameter, degree=degree, knots=knots, normalize=normalize_parameter
    )
    for index, (point, weight) in enumerate(point_weights):
        blended_matrix.weight_matrix[index].weight_in.set(weight)
        blended_matrix.weight_matrix[index].matrix_in.connect_from(point)

    blended_matrix_attribute = blended_matrix.matrix_sum

    if reset_transforms:
        zero_transform(pinned_transform)

    # If there's no fancy interpolation we can just regularize the matrix and early out.
    if not (interpolate_scale or align_tangent):
        output_matrix = _create_pick_matrix(
            segment_name=segment_name,
            input_matrix=blended_matrix_attribute,
            interpolate_rotation=interpolate_rotation,
        )
        output_matrix.output_matrix.connect_to(f"{pinned_transform}.offsetParentMatrix")
        matrix_spline.pinned_transforms.append(pinned_transform)
        return

    # Create nodes to access the values of the blended matrix node.
    blended_matrix_row1 = node.RowFromMatrixNode.create(name=f"{blended_matrix}_row1")
    blended_matrix_row1.input.set(0)
    blended_matrix_row1.matrix.connect_from(blended_matrix_attribute)

    blended_matrix_row2 = node.RowFromMatrixNode.create(name=f"{blended_matrix}_row2")
    blended_matrix_row2.input.set(1)
    blended_matrix_row2.matrix.connect_from(blended_matrix_attribute)

    blended_matrix_row3 = node.RowFromMatrixNode.create(name=f"{blended_matrix}_row3")
    blended_matrix_row3.input.set(2)
    blended_matrix_row3.matrix.connect_from(blended_matrix_attribute)

    blended_matrix_row4 = node.RowFromMatrixNode.create(name=f"{blended_matrix}_row4")
    blended_matrix_row4.input.set(3)
    blended_matrix_row4.matrix.connect_from(blended_matrix_attribute)

    axis_to_row: dict[tuple[int, int, int], node.RowFromMatrixNode] = {
        (1, 0, 0): blended_matrix_row1,
        (0, 1, 0): blended_matrix_row2,
        (0, 0, 1): blended_matrix_row3,
        (-1, 0, 0): blended_matrix_row1,
        (0, -1, 0): blended_matrix_row2,
        (0, 0, -1): blended_matrix_row3,
    }

    rigid_matrix_output: MatrixAttribute
    if align_tangent:
        rigid_matrix, rigid_matrix_output, tangent_vector = _create_align_tangent(
            segment_name=segment_name,
            cv_matrices=cv_matrices,
            parameter=parameter,
            degree=degree,
            knots=knots,
            normalize_parameter=normalize_parameter,
            primary_axis=primary_axis,
            secondary_axis=secondary_axis,
            twist=twist,
            axis_to_row=axis_to_row,
        )
    else:
        rigid_matrix = _create_pick_matrix(
            segment_name=segment_name,
            input_matrix=blended_matrix_attribute,
            interpolate_rotation=interpolate_rotation,
        )
        rigid_matrix_output = rigid_matrix.output_matrix

    tangent_scale_attr: ScalarAttribute | None = None
    if align_tangent and stretch and tangent_vector is not None:
        tangent_scale_attr = _create_tangent_scale(segment_name, tangent_vector)

    # Create nodes to access the values of the rigid matrix (aim matrix or pick matrix) node.
    rigid_matrix_row1 = node.RowFromMatrixNode.create(name=f"{rigid_matrix}_row1")
    rigid_matrix_row1.matrix.connect_from(rigid_matrix_output)
    rigid_matrix_row1.input.set(0)

    rigid_matrix_row2 = node.RowFromMatrixNode.create(name=f"{rigid_matrix}_row2")
    rigid_matrix_row2.matrix.connect_from(rigid_matrix_output)
    rigid_matrix_row2.input.set(1)

    rigid_matrix_row3 = node.RowFromMatrixNode.create(name=f"{rigid_matrix}_row3")
    rigid_matrix_row3.matrix.connect_from(rigid_matrix_output)
    rigid_matrix_row3.input.set(2)

    # Create Nodes to re-apply scale

    x_scaled = _scale_vector(
        node_name=f"{segment_name}_x_scale",
        vector_attr=rigid_matrix_row1.output,
        scale_attr=blended_matrix_row1.output.w,
        axis=X_AXIS,
        stretch=stretch,
        tangent_scale_attr=tangent_scale_attr,
        primary_axis=primary_axis,
        interpolate_scale=interpolate_scale,
    )

    y_scaled = _scale_vector(
        node_name=f"{segment_name}_y_scale",
        vector_attr=rigid_matrix_row2.output,
        scale_attr=blended_matrix_row2.output.w,
        axis=Y_AXIS,
        stretch=stretch,
        tangent_scale_attr=tangent_scale_attr,
        primary_axis=primary_axis,
        interpolate_scale=interpolate_scale,
    )

    z_scaled = _scale_vector(
        node_name=f"{segment_name}_z_scale",
        vector_attr=rigid_matrix_row3.output,
        scale_attr=blended_matrix_row3.output.w,
        axis=Z_AXIS,
        stretch=stretch,
        tangent_scale_attr=tangent_scale_attr,
        primary_axis=primary_axis,
        interpolate_scale=interpolate_scale,
    )

    # Rebuild the matrix
    output_matrix = node.FourByFourMatrixNode.create(name=f"{segment_name}_output_matrix")
    x_scaled.x.connect_to(output_matrix.in_00)
    x_scaled.y.connect_to(output_matrix.in_01)
    x_scaled.z.connect_to(output_matrix.in_02)

    y_scaled.x.connect_to(output_matrix.in_10)
    y_scaled.y.connect_to(output_matrix.in_11)
    y_scaled.z.connect_to(output_matrix.in_12)

    z_scaled.x.connect_to(output_matrix.in_20)
    z_scaled.y.connect_to(output_matrix.in_21)
    z_scaled.z.connect_to(output_matrix.in_22)

    blended_matrix_row4.output.x.connect_to(output_matrix.in_30)
    blended_matrix_row4.output.y.connect_to(output_matrix.in_31)
    blended_matrix_row4.output.z.connect_to(output_matrix.in_32)

    output_matrix.output.connect_to(f"{pinned_transform}.offsetParentMatrix")
    matrix_spline.pinned_transforms.append(pinned_transform)


def pin_transforms_to_matrix_spline(
    matrix_spline: MatrixSpline,
    pinned_transforms: Sequence[str],
    parameters: Sequence[float] | None = None,
    padded: bool = True,
    stretch: bool = True,
    arc_length: bool = True,
    primary_axis: tuple[int, int, int] | None = (0, 1, 0),
    secondary_axis: tuple[int, int, int] | None = (0, 0, 1),
    twist: bool = True,
    align_tangent: bool = True,
    interpolate_rotation: bool = True,
    interpolate_scale: bool = True,
    u_start: float | None = None,
    u_end: float | None = None,
) -> MatrixSpline:
    """
    Takes a set of transforms pins them to a MatrixSpline (note that the pins are all calculated in local space!).
    Args:
        matrix_spline: The matrix spline defention that will drive the pinned transforms.
        pinned_transforms: These transforms will be constrained to the spline.
        parameters: Optional manual specification of the paramters to pin to.
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
        interpolate_rotation: When True the rotation of the pinned transform will be interpolated with the CVs rotations.
        interpolate_scale: When True the scale of the pinned transform will be a spline interpolation of the CVs scales.
        u_start: Optional start parameter for the section to which transforms will be pinned.
        u_end: Optional end parameter for the section to which transforms will be pinned.
    Returns:
        matrix_spline: The matrix spline.
    """
    segments = len(pinned_transforms)
    segment_parameters: Sequence[float]
    if parameters is not None:
        segment_parameters = parameters
    else:
        cv_positions: list[Vector3] = []
        for transform in matrix_spline.cv_transforms:
            position: tuple[float, float, float] = cmds.xform(  # type: ignore
                transform, query=True, worldSpace=True, translation=True
            )
            cv_positions.append(Vector3(*position))
        segment_parameters = resample(
            cv_positions=cv_positions,
            number_of_points=segments,
            degree=matrix_spline.degree,
            knots=matrix_spline.knots,
            periodic=matrix_spline.periodic,
            padded=padded,
            arc_length=arc_length,
            normalize_parameter=False,
            u_start=u_start,
            u_end=u_end,
        )

    for transform, parameter in zip(pinned_transforms, segment_parameters, strict=True):
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
            interpolate_rotation=interpolate_rotation,
            interpolate_scale=interpolate_scale,
        )
    return matrix_spline
