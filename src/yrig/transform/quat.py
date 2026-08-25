from yrig.maya_api.attribute import QuatAttribute
from yrig.maya_api.node import (
    QuatInvertNode,
    QuatNormalizeNode,
    QuatProdNode,
    QuatToEulerNode,
)
from yrig.name import get_short_name
from yrig.transform.matrix import localize_and_decompose_matrix, matrix_constraint
from yrig.transform.structs import Axis
from yrig.transform.utils import create_transform, match_transform


def drive_rotation_with_quat(transform: str, quat_attribute: QuatAttribute) -> None:
    """
    Drive the rotation of a transform using a QuatAttribute via a QuatToEuler node.
    """
    euler_angles = QuatToEulerNode.create(f"{get_short_name(transform)}_euler")
    euler_angles.input_rotate_order.connect_from(f"{transform}.rotateOrder")
    quat_attribute.connect_to(euler_angles.input_quat)
    euler_angles.output_rotate.connect_to(f"{transform}.rotate")


def _connect_twist_quat(source: QuatAttribute, destination: QuatAttribute, axis: Axis) -> None:
    """
    Connect the W and the specified axis component from one quaternion to another.
    (Forming a twist Decomposition) https://www.chadvernon.com/blog/swing-twist/
    """
    source.w.connect_to(destination.w)
    if axis == "x":
        source.x.connect_to(destination.x)
    elif axis == "y":
        source.y.connect_to(destination.y)
    elif axis == "z":
        source.z.connect_to(destination.z)


def twist_extract_euler(transform: str, reference_space: str, axis: Axis) -> QuatToEulerNode:
    """
    Extract the twist of a transform relative to a specified axis and
    output it as Euler angles.
    Args:
        transform: Name of the transform whose twist should be extracted.
        reference_space: Transform used as the reference space for the twist.
        axis: Axis around which the twist should be isolated ("x", "y", or "z").

    Returns:
        QuatToEulerNode: Node producing the extracted twist as Euler rotation.
    """
    name = f"{get_short_name(transform)}_twist"
    decompose = localize_and_decompose_matrix(transform, reference_space)
    euler_output = QuatToEulerNode.create(f"{name}_euler")
    _connect_twist_quat(decompose.output_quat, euler_output.input_quat, axis)
    return euler_output


def twist_extract_quat(transform: str, reference_space: str, axis: Axis) -> QuatAttribute:
    """
    Extract the twist of a transform relative to a specified axis and
    output it as a normalized quaternion.
    Args:
        transform: Name of the transform whose twist should be extracted.
        reference_space: Transform used as the reference space for the twist.
        axis: Axis around which the twist should be isolated ("x", "y", or "z").

    Returns:
        QuatAttribute: The attribute with the resulting twist calculation.
    """
    name = f"{get_short_name(transform)}_twist"
    decompose = localize_and_decompose_matrix(transform, reference_space)
    output = QuatNormalizeNode.create(name)
    _connect_twist_quat(decompose.output_quat, output.input_quat, axis)
    return output.output_quat


def swing_extract_quat(transform: str, reference_space: str, axis: Axis) -> QuatAttribute:
    """
    Extract the swing of a transform relative to a specified axis and
    output it as a normalized quaternion.
    Args:
        transform: Name of the transform whose swing should be extracted.
        reference_space: Transform used as the reference space for the swing.
        axis: Axis around which the twist should be isolated ("x", "y", or "z").

    Returns:
        QuatAttribute: The attribute with the resulting swing calculation.
    """
    name = f"{get_short_name(transform)}_swing"
    decompose = localize_and_decompose_matrix(transform, reference_space)
    inverse = QuatInvertNode.create(f"{get_short_name(transform)}_twist_inverse")
    _connect_twist_quat(decompose.output_quat, inverse.input_quat, axis)
    swing = QuatProdNode.create(name)
    inverse.output_quat.connect_to(swing.input1_quat)
    decompose.output_quat.connect_to(swing.input2_quat)
    return swing.output_quat


def create_swing_only_transform(
    transform: str,
    reference_space: str,
    axis: Axis,
    name: str | None = None,
    parent: str | None = None,
    constrain_translate: bool = True,
) -> str:
    """
    Extracts the swing of a transform relative to a specific space and specified axis and
    creates a transform that follows only the position and swing.
    Args:
        transform: Name of the transform whose swing should be extracted.
        reference_space: Transform used as the reference space for the swing.
        axis: Axis around which the twist should be isolated ("x", "y", or "z").
        name (optional): The name for the newly created swing transform
        parent (optional): Parent for the newly created swing transform. If None, will be set to the reference space.
        constrain_translate: If True the swing transform will follow the translation of the source transform along with its swing.

    Returns:
        str: The name of the created swing transform.
    """
    used_name = name if name is not None else f"{get_short_name(transform)}_swing"
    used_parent = parent if parent is not None else reference_space

    neutral_group = create_transform(name=f"{name}_npo", parent=used_parent)
    swing_transform = create_transform(name=used_name, parent=neutral_group)
    match_transform(neutral_group, transform)
    if parent is not None:
        matrix_constraint(reference_space, neutral_group)
    swing_quat = swing_extract_quat(transform, neutral_group, axis)
    if constrain_translate:
        matrix_constraint(
            transform,
            swing_transform,
            translate=True,
            rotate=False,
            scale=False,
            shear=False,
        )
    drive_rotation_with_quat(swing_transform, swing_quat)
    return swing_transform
