import logging
from collections.abc import Iterable, Sequence
from typing import TypeAlias

from maya import cmds
from maya.api.OpenMaya import (
    MAngle,
    MEulerRotation,
    MFnDependencyNode,
    MFnTransform,
    MMatrix,
    MPlug,
    MSpace,
    MTransformationMatrix,
)

from yrig.maya_api import node
from yrig.maya_api.attribute import MatrixAttribute, Vector3Attribute
from yrig.maya_api.node import (
    DecomposeMatrixNode,
    MultMatrixNode,
)
from yrig.maya_api.utils import get_dag_path, get_depend_node
from yrig.name import get_short_name

log = logging.getLogger(__name__)

# fmt: off
MatrixTuple: TypeAlias = tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]
# fmt: on


def is_identity_matrix(
    matrix: MMatrix | MatrixTuple | Sequence[float], epsilon: float = 0.001
) -> bool:
    """Check whether a 4×4 matrix is approximately equal to the identity matrix.

    Compares each element of *matrix* against the corresponding element of
    the identity matrix using the given *epsilon* tolerance.

    Args:
        matrix: The matrix to test, supplied as an ``MMatrix``, a 16-element
            tuple, or any sequence of 16 floats in row-major order.
        epsilon: Maximum per-element deviation from identity that is still
            considered equivalent.  Defaults to ``0.001``.

    Returns:
        ``True`` if every element is within *epsilon* of the identity value,
        ``False`` otherwise.
    """
    if isinstance(matrix, MMatrix):
        return matrix.isEquivalent(MMatrix.kIdentity, epsilon)
    return all(
        abs(value - identity) < epsilon
        for value, identity in zip(
            matrix, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], strict=True
        )
    )


def mmatrix_to_list(matrix: MMatrix) -> list[float]:
    """Flatten an ``MMatrix`` into a row-major list of 16 floats.

    Args:
        matrix: The Maya ``MMatrix`` to convert.

    Returns:
        A list of 16 ``float`` values in row-major order
        (row 0 cols 0–3, row 1 cols 0–3, …).
    """
    return [matrix.getElement(row, col) for row in range(4) for col in range(4)]


def get_local_matrix(transform: str) -> MMatrix:
    """
    Returns the local matrix of a transform.
    """
    dag_path = get_dag_path(transform)
    mfn_transform: MFnTransform = MFnTransform(dag_path)
    transformation: MTransformationMatrix = mfn_transform.transformation()
    return transformation.asMatrix()


def get_parent_matrix(transform: str) -> MMatrix:
    """
    Returns the full world matrix of a transform up to its parent, including rotateAxis, jointOrient, etc.
    Equivalent to Maya's internal parent matrix.
    """
    dag_path = get_dag_path(transform)
    return dag_path.exclusiveMatrix()


def get_parent_inverse_matrix(transform: str) -> MMatrix:
    """
    Returns the full inverse world matrix of a transform up to its parent, including rotateAxis, jointOrient, etc.
    Equivalent to Maya's internal parentInverse matrix.
    """
    dag_path = get_dag_path(transform)
    return dag_path.exclusiveMatrixInverse()


def get_world_matrix(transform: str) -> MMatrix:
    """
    Returns the full world matrix of a transform, including rotateAxis, jointOrient, etc.
    Equivalent to Maya's internal world matrix.
    """
    dag_path = get_dag_path(transform)
    return dag_path.inclusiveMatrix()


def set_local_matrix(
    transform: str, matrix: MMatrix, use_joint_orient: bool = True, fallback: bool = False
) -> None:
    """Set the local transformation of a Maya transform node from a matrix.

    Decomposes the given matrix into translate, rotate, scale, and shear
    components and applies them to the node's local channels.  For joint
    nodes the ``jointOrient`` attribute is zeroed so that the full
    orientation lives in the rotate channels.

    Args:
        transform: The name of the Maya transform (or joint) node to
            modify.
        matrix: The desired local-space matrix.
        use_joint_orient: When ``True``, if the transform is a joint the rotation will be zeroed and applied to the jointOrient.
        fallback: When ``True``, use ``cmds.xform`` to set the matrix in
            one call instead of decomposing it into individual channels.
            This is less precise but can be useful as a workaround for
            edge-case node types.
    """
    if fallback:
        cmds.xform(transform, worldSpace=False, matrix=matrix)  # type: ignore
    else:
        # Apply local matrix using transformation matrix
        transform_matrix: MTransformationMatrix = MTransformationMatrix(matrix)

        # Set translation
        translation = transform_matrix.translation(MSpace.kTransform)
        cmds.setAttr(f"{transform}.translate", translation.x, translation.y, translation.z)
        node_type = cmds.nodeType(transform)

        rotate_order = cmds.getAttr(f"{transform}.rotateOrder")
        xyz_rotation = transform_matrix.rotation()  # xyz rotation for jointOrient case

        # Reorder for this transform's rotateOrder
        transform_matrix.reorderRotation(rotate_order + 1)
        rotation = transform_matrix.rotation()

        rotation_deg = (
            MAngle(rotation.x).asDegrees(),
            MAngle(rotation.y).asDegrees(),
            MAngle(rotation.z).asDegrees(),
        )

        xyz_rotation_deg = (
            MAngle(xyz_rotation.x).asDegrees(),
            MAngle(xyz_rotation.y).asDegrees(),
            MAngle(xyz_rotation.z).asDegrees(),
        )

        # Set rotation
        if node_type == "joint":  # TODO: figure out how to handle segmentScaleCompensate
            if use_joint_orient:
                cmds.setAttr(f"{transform}.rotate", 0, 0, 0)  # type: ignore
                cmds.setAttr(f"{transform}.jointOrient", *xyz_rotation_deg)
            else:
                cmds.setAttr(f"{transform}.jointOrient", 0, 0, 0)  # type: ignore
                cmds.setAttr(f"{transform}.rotate", *rotation_deg)
        else:
            cmds.setAttr(f"{transform}.rotate", *rotation_deg)

        # Set scale
        scale = transform_matrix.scale(MSpace.kTransform)
        cmds.setAttr(f"{transform}.scale", scale[0], scale[1], scale[2])

        # Set shear
        shear = transform_matrix.shear(MSpace.kTransform)
        cmds.setAttr(f"{transform}.shear", shear[0], shear[1], shear[2])


def set_world_matrix(
    transform: str, matrix: MMatrix, use_joint_orient: bool = True, fallback: bool = False
) -> None:
    """Set the world-space matrix of a transform by converting to local space first.

    The given world matrix is multiplied by the parent's inverse world
    matrix to obtain a local matrix, which is then applied via `set_local_matrix`.

    Args:
        transform: Maya transform node name.
        matrix: Target world space matrix.
        use_joint_orient: When ``True``, if the transform is a joint the rotation will be zeroed and applied to the jointOrient.
        fallback: If True, use cmds.xform instead of manual decomposition.
    """
    if fallback:
        cmds.xform(transform, worldSpace=True, matrix=matrix)  # type: ignore
    else:
        inverse_matrix: MMatrix = get_parent_inverse_matrix(transform)
        local_matrix: MMatrix = matrix * inverse_matrix
        set_local_matrix(transform, local_matrix, use_joint_orient)


def multiply_matrices(
    name: str,
    matrices: Iterable[MatrixAttribute | str | MMatrix | Sequence[float]],
    skip_identity_matrices: bool = True,
) -> MultMatrixNode:
    """
    Create a ``multMatrix`` node that multiplies the given matrices in input order.
    Attributes and attribute paths are connected. Matrix values are assigned directly.
    """
    mult_matrix_node = MultMatrixNode.create(name)
    index: int = 0
    for matrix in matrices:
        if isinstance(matrix, (MatrixAttribute, str)):
            mult_matrix_node.matrix_in[index].connect_from(matrix)
        else:
            if skip_identity_matrices and is_identity_matrix(matrix):
                continue
            mult_matrix_node.matrix_in[index].set(matrix)
        index += 1
    return mult_matrix_node


def localize_world_matrix(transform: str, target_space_transform: str) -> MultMatrixNode:
    """Create a multMatrix node localizing transform into target_space_transform's space."""
    localize_matrix = node.MultMatrixNode.create(
        f"{get_short_name(target_space_transform)}_local_{get_short_name(transform)}"
    )
    localize_matrix.matrix_in[0].connect_from(f"{transform}.worldMatrix[0]")
    localize_matrix.matrix_in[1].connect_from(f"{target_space_transform}.worldInverseMatrix[0]")
    return localize_matrix


def localize_and_decompose_matrix(transform: str, parent: str) -> DecomposeMatrixNode:
    """Create a network that localizing transform into target_space_transform's space and connect it to a new decomposeMatrix node."""
    localize_matrix = localize_world_matrix(transform, parent)
    decompose = DecomposeMatrixNode.create(f"{get_short_name(transform)}_decompose")
    localize_matrix.matrix_sum.connect_to(decompose.input_matrix)
    return decompose


def _get_joint_orient_euler(joint: str) -> MEulerRotation:
    obj = get_depend_node(joint)
    fn: MFnDependencyNode = MFnDependencyNode(obj)
    plug: MPlug = fn.findPlug("jointOrient", False)
    # Radians
    x = plug.child(0).asDouble()
    y = plug.child(1).asDouble()
    z = plug.child(2).asDouble()
    return MEulerRotation(x, y, z)


def drive_transform_with_matrix(
    matrix_attr: MatrixAttribute | str,
    transform: str,
    translate: bool = True,
    rotate: bool = True,
    scale: bool = True,
    shear: bool = True,
    use_joint_orient: bool = False,
    lock_joint_orient: bool = True,
) -> None:
    """
    Drive a transforms translate rotate scale and shear with a matrix attribute.

    Args:
        matrix_attr: The matrix attribute to use as the driver.
        transform: The transform to be driven.
        translate: whether to constrain translation.
        use_joint_orient: when true the joint orient is taken into account, otherwise it is set to zero.
        lock_joint_orient: When True, if the transform is a joint
            it's joint orient will be locked after being zeroed to keep maya from screwing it up later when re-parenting.
    """
    constraint_name: str = get_short_name(transform)

    # Create the decomposed matrix and connect it's inputs
    decompose_matrix = DecomposeMatrixNode.create(f"{constraint_name}_driver_decompose")
    decompose_matrix.input_matrix.connect_from(matrix_attr)
    decompose_matrix.input_rotate_order.connect_from(f"{transform}.rotateOrder")

    rotate_attr: Vector3Attribute = decompose_matrix.output_rotate
    # Drive transform with decomposed values
    # If it's a joint we have to do a whole bunch of other nonsense to account for joint orient
    if cmds.nodeType(transform) == "joint":
        if scale:
            cmds.setAttr(f"{transform}.segmentScaleCompensate", 0)  # type: ignore
        if rotate:
            if use_joint_orient:
                # We need to handle Joint Orient for Pose Driver, IK and other Maya features.
                # https://help.autodesk.com/cloudhelp/ENU/MayaCRE-Tech-Docs/Nodes/joint.html
                # joint matrix = scale * rotateAxis * rotate * jointOrient * parentScaleInverse * translate
                # So we need to compensate for jointOrient by multiplying our final rotation by the inverse of the jointOrient

                # Turns out we don't need to specifically isolate the rotation before doing the matrix multiplication
                # rotation_matrix = PickMatrixNode(f"{constraint_name}_rotation")
                # rotation_matrix.input_matrix.connect_from(matrix_attr)

                joint_orient_matrix: MMatrix = _get_joint_orient_euler(transform).asMatrix()
                # Only add the compensation if needed.
                if not is_identity_matrix(matrix=joint_orient_matrix):
                    joint_orient_matrix_inverse: MMatrix = joint_orient_matrix.inverse()
                    rotation_mult = MultMatrixNode.create(f"{constraint_name}_oriented_rotation")
                    rotation_mult.matrix_in[0].connect_from(matrix_attr)
                    rotation_mult.matrix_in[1].set(joint_orient_matrix_inverse)
                    orient_matrix_decompose = DecomposeMatrixNode.create(
                        f"{constraint_name}_orient_decompose"
                    )
                    orient_matrix_decompose.input_matrix.connect_from(rotation_mult.matrix_sum)
                    orient_matrix_decompose.input_rotate_order.connect_from(
                        f"{transform}.rotateOrder"
                    )
                    rotate_attr = orient_matrix_decompose.output_rotate
            else:
                cmds.setAttr(f"{transform}.jointOrient", lock=False)
                cmds.setAttr(f"{transform}.jointOrient", 0, 0, 0, type="float3")  # type: ignore
            if lock_joint_orient:
                cmds.setAttr(f"{transform}.jointOrient", lock=True)
    # Connect each channel individually as if they are connected from the parent compound attr certain things break,
    # such as pose interpolator goToPose command. Native Maya constraints also connect this way.
    if translate:
        decompose_matrix.output_translate.x.connect_to(f"{transform}.translateX")
        decompose_matrix.output_translate.y.connect_to(f"{transform}.translateY")
        decompose_matrix.output_translate.z.connect_to(f"{transform}.translateZ")
    if rotate:
        rotate_attr.x.connect_to(f"{transform}.rotateX")
        rotate_attr.y.connect_to(f"{transform}.rotateY")
        rotate_attr.z.connect_to(f"{transform}.rotateZ")
        cmds.setAttr(f"{transform}.rotateAxis", 0, 0, 0, type="float3")  # type: ignore
    if scale:
        decompose_matrix.output_scale.x.connect_to(f"{transform}.scaleX")
        decompose_matrix.output_scale.y.connect_to(f"{transform}.scaleY")
        decompose_matrix.output_scale.z.connect_to(f"{transform}.scaleZ")
    if shear:
        decompose_matrix.output_shear.x.connect_to(f"{transform}.shearXY")
        decompose_matrix.output_shear.y.connect_to(f"{transform}.shearXZ")
        decompose_matrix.output_shear.z.connect_to(f"{transform}.shearYZ")
