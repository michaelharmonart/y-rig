from typing import Sequence, TypeAlias

import maya.cmds as cmds
from maya.api.OpenMaya import (
    MAngle,
    MDagPath,
    MFnTransform,
    MMatrix,
    MSelectionList,
    MSpace,
    MTransformationMatrix,
)

from yrig.maya_api import node

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
    if isinstance(matrix, MMatrix):
        return matrix.isEquivalent(MMatrix.kIdentity, epsilon)
    return all(
        abs(value - identity) < epsilon
        for value, identity in zip(matrix, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    )


def mmatrix_to_list(matrix: MMatrix) -> list[float]:
    return [matrix.getElement(row, col) for row in range(4) for col in range(4)]


def get_local_matrix(transform: str) -> MMatrix:
    """
    Returns the local matrix of a transform.
    """
    selection = MSelectionList()
    selection.add(transform)
    dag_path: MDagPath = selection.getDagPath(0)
    mfn_transform: MFnTransform = MFnTransform(dag_path)
    transformation: MTransformationMatrix = mfn_transform.transformation()
    return transformation.asMatrix()


def get_parent_matrix(transform: str) -> MMatrix:
    """
    Returns the full world matrix of a transform up to it's parent, including rotateAxis, jointOrient, etc.
    Equivalent to Maya's internal parent matrix.
    """
    selection = MSelectionList()
    selection.add(transform)
    dag_path: MDagPath = selection.getDagPath(0)
    return dag_path.exclusiveMatrix()


def get_parent_inverse_matrix(transform: str) -> MMatrix:
    """
    Returns the full inverse world matrix of a transform up to it's parent, including rotateAxis, jointOrient, etc.
    Equivalent to Maya's internal parentInverse matrix.
    """
    selection = MSelectionList()
    selection.add(transform)
    dag_path: MDagPath = selection.getDagPath(0)
    return dag_path.exclusiveMatrixInverse()


def get_world_matrix(transform: str) -> MMatrix:
    """
    Returns the full world matrix of a transform, including rotateAxis, jointOrient, etc.
    Equivalent to Maya's internal world matrix.
    """
    selection = MSelectionList()
    selection.add(transform)
    dag_path: MDagPath = selection.getDagPath(0)
    return dag_path.inclusiveMatrix()


def set_world_matrix(transform: str, matrix: MMatrix, fallback=False) -> None:
    """
    Set the world matrix of a transform by decomposing it into local components.

    Args:
        transform: Maya transform node name.
        matrix: Target world space matrix.
        fallback: If True, use cmds.xform instead of manual decomposition.
    """
    if fallback:
        cmds.xform(transform, worldSpace=True, matrix=matrix)  # type: ignore
    else:
        inverse_matrix: MMatrix = get_parent_inverse_matrix(transform)
        local_matrix: MMatrix = matrix * inverse_matrix

        # Apply local matrix using transformation matrix
        transform_matrix: MTransformationMatrix = MTransformationMatrix(local_matrix)
        # Set translation
        translation = transform_matrix.translation(MSpace.kTransform)
        cmds.setAttr(f"{transform}.translate", translation.x, translation.y, translation.z)
        node_type = cmds.nodeType(transform)

        rotate_order = cmds.getAttr(f"{transform}.rotateOrder")
        transform_matrix.reorderRotation(rotate_order + 1)
        rotation = transform_matrix.rotation()
        cmds.setAttr(
            f"{transform}.rotate",
            MAngle(rotation.x).asDegrees(),
            MAngle(rotation.y).asDegrees(),
            MAngle(rotation.z).asDegrees(),
        )

        if node_type == "joint":
            # Zero the rotate channel
            cmds.setAttr(f"{transform}.jointOrient", 0, 0, 0)  # type: ignore

        # Set scale
        scale = transform_matrix.scale(MSpace.kTransform)
        cmds.setAttr(f"{transform}.scale", scale[0], scale[1], scale[2])

        # Set shear
        shear = transform_matrix.shear(MSpace.kTransform)
        cmds.setAttr(f"{transform}.shear", shear[0], shear[1], shear[2])


def matrix_constraint(
    source_transform: str,
    constrain_transform: str,
    keep_offset: bool = True,
    local_space: bool = True,
    use_joint_orient: bool = False,
    translate: bool = True,
    rotate: bool = True,
    scale: bool = True,
    shear: bool = True,
) -> None:
    """
    Constrain a transform to another.

    Args:
        source_transform: joint to match.
        constrain_joint: joint to constrain.
        keep_offset: keep the offset of the constrained transform to the source at time of constraint generation.
        local_space: if False the constrained transform will have inheritsTransform turned off.
        use_joint_orient: when true the joint orient is taken into account, otherwise it is set to zero.
        translate: whether to constrain translation.
        rotate: whether to constrain rotation.
        scale: whether to constrain scale.
        shear: whether to constrain shear.
    """
    constraint_name: str = constrain_transform.split("|")[-1]

    # Create node to multiply matrices, as well as a counter to make sure to input into the right slot.
    mult_index: int = 0
    mult_matrix = node.MultMatrixNode(name=f"{constraint_name}_ConstraintMatrixMult")

    # If we want to keep the offset, we put the position of the constrained transform into
    # the source transform's space and record it.
    if keep_offset:
        # Get the offset matrix
        offset_matrix: MMatrix = (
            get_world_matrix(constrain_transform) * get_world_matrix(source_transform).inverse()
        )

        # Check the matrix against an identity matrix. If it's the same within a margin of error,
        # the transforms aren't offset, meaning we can skip that extra matrix multiplication.
        if not is_identity_matrix(matrix=offset_matrix):
            # Put the offset into the matrix multiplier
            mult_matrix.matrix_in[mult_index].set(offset_matrix)
            mult_index += 1
        else:
            keep_offset = False

    # Next we multiply by the world matrix of the source transform
    mult_matrix.matrix_in[mult_index].connect_from(f"{source_transform}.worldMatrix[0]")
    mult_index += 1

    # If we have a parent transform we then put it into that space by multiplying by it's worldInverseMatrix
    if local_space:
        mult_matrix.matrix_in[mult_index].connect_from(
            f"{constrain_transform}.parentInverseMatrix[0]"
        )
        mult_index += 1
    else:
        cmds.setAttr(f"{constrain_transform}.inheritsTransform", 0)  # type: ignore

    # Create the decomposed matrix and connect it's inputs
    decompose_matrix = node.DecomposeMatrixNode(f"{constraint_name}_ConstrainMatrixDecompose")
    mult_matrix.matrix_sum.connect_to(decompose_matrix.input_matrix)
    decompose_matrix.input_rotate_order.connect_from(f"{constrain_transform}.rotateOrder")

    rotate_attr = decompose_matrix.output_rotate
    # If it's a joint we have to do a whole bunch of other nonsense to account for joint orient (I was up till 2am because of this)
    if cmds.nodeType(constrain_transform) == "joint":
        if scale:
            cmds.setAttr(f"{constrain_transform}.segmentScaleCompensate", 0)  # type: ignore
        if rotate:
            if use_joint_orient:
                # Check if the joint orient isn't about 0
                joint_orient: tuple[float, float, float] = cmds.getAttr(
                    f"{constrain_transform}.jointOrient"
                )[0]
                if any(abs(i) > 0.01 for i in joint_orient):
                    # Get our joint orient and turn it into a matrix
                    orient_node: str = cmds.createNode(
                        "composeMatrix", name=f"{constraint_name}_OrientMatrix"
                    )
                    cmds.connectAttr(
                        f"{constrain_transform}.jointOrient", f"{orient_node}.inputRotate"
                    )
                    orient_matrix = cmds.getAttr(f"{orient_node}.outputMatrix")

                    # We need to compose a different matrix to drive just the rotation due to the joint orient
                    orient_offset_node = node.InverseMatrixNode(
                        name=f"{constraint_name}_OrientOffsetMatrix"
                    )
                    orient_mult_matrix = node.MultMatrixNode(
                        name=f"{constraint_name}_ConstraintOrientMatrix"
                    )
                    orient_mult_index: int = 0

                    # If we have an offset it'll be our first matrix in the multiplier (same as above)
                    if keep_offset:
                        orient_mult_matrix.matrix_in[orient_mult_index].set(offset_matrix)  # type: ignore
                        orient_mult_index += 1

                    # Next we multiply by the world matrix of the source transform
                    orient_mult_matrix.matrix_in[orient_mult_index].connect_from(
                        f"{source_transform}.worldMatrix[0]"
                    )
                    orient_mult_index += 1

                    # Depending on if we need to take a parent into account we'll need a few extra nodes
                    # (otherwise just pre-calculate a matrix and plop it in)
                    # Bless Jared Love for figuring this out https://www.youtube.com/watch?v=_LNhZB8jQyo
                    # Essentially we need to take the inverse of the orient * the world matrix of the parent and multiply by that
                    if local_space:
                        # Create a node to multiply the joint orient by the world matrix of the parent
                        orient_parent_mult_matrix = node.MultMatrixNode(
                            name=f"{constraint_name}_ConstraintOrientMultMatrix"
                        )
                        orient_parent_mult_matrix.matrix_in[0].set(orient_matrix)
                        orient_parent_mult_matrix.matrix_in[1].connect_from(
                            f"{constrain_transform}.parentMatrix[0]"
                        )

                        # Create an inverse node and connect it to the result of the last step
                        orient_parent_mult_matrix.matrix_sum.connect_to(
                            orient_offset_node.input_matrix
                        )

                        # Finally add this to a slot on the matrix multiplier node
                        orient_offset_node.output_matrix.connect_to(
                            orient_mult_matrix.matrix_in[orient_mult_index]
                        )
                        orient_mult_index += 1
                    else:
                        # If we don't care about a parent, just make a temp inverse node and store the inverse of the joint orient
                        orient_offset_node.input_matrix.connect_from(f"{orient_node}.outputMatrix")

                        inverse_orient_matrix = orient_offset_node.output_matrix.get()

                        # And then set it in a slot on the matrix multiplier
                        orient_mult_matrix.matrix_in[orient_mult_index].set(inverse_orient_matrix)
                        orient_mult_index += 1
                        # Cleanup temp node
                        orient_offset_node.delete()

                    #  Hook up the matrix multiplier to our decomposeMatrix and feed it into the rotate attribute of the joint
                    orient_decompose_matrix = node.DecomposeMatrixNode(
                        name=f"{constraint_name}_ConstrainOrientDecompose"
                    )
                    orient_mult_matrix.matrix_sum.connect_to(orient_decompose_matrix.input_matrix)
                    orient_decompose_matrix.input_rotate_order.connect_from(
                        f"{constrain_transform}.rotateOrder"
                    )
                    rotate_attr = orient_decompose_matrix.output_rotate
            else:
                cmds.setAttr(f"{constrain_transform}.jointOrient", 0, 0, 0, type="float3")  # type: ignore

    if translate:
        decompose_matrix.output_translate.connect_to(f"{constrain_transform}.translate")
    if rotate:
        cmds.setAttr(f"{constrain_transform}.rotateAxis", 0, 0, 0, type="float3")  # type: ignore
        rotate_attr.connect_to(f"{constrain_transform}.rotate")
    if scale:
        decompose_matrix.output_scale.connect_to(f"{constrain_transform}.scale")
    if shear:
        decompose_matrix.output_shear.connect_to(f"{constrain_transform}.shear")
