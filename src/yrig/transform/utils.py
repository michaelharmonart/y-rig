import maya.cmds as cmds
from maya.api.OpenMaya import MAngle, MEulerRotation, MMatrix, MPoint

from yrig.transform.matrix import (
    get_world_matrix,
    set_local_matrix,
    set_world_matrix,
)


def get_shapes(transform: str) -> list[str]:
    # list the shapes of node
    shape_list: list[str] = cmds.listRelatives(
        transform, shapes=True, noIntermediate=True, children=True
    )

    if shape_list:
        return shape_list
    else:
        raise RuntimeError(f"{transform} has no child shape nodes")


def get_position(transform: str, world_space: bool = True) -> MPoint:
    return MPoint(cmds.xform(transform, query=True, worldSpace=world_space, translation=True))


def match_transform(transform: str, target_transform: str) -> None:
    """
    Match a transform to another in world space.

    Args:
        transform: Object to be moved to the specified transform.
        target_transform: Name of the transform to match to.
    """
    source_matrix: MMatrix = get_world_matrix(transform=target_transform)
    set_world_matrix(transform=transform, matrix=source_matrix)


def match_location(transform: str, target_transform: str) -> None:
    """
    Match a transforms location to another in world space.

    Args:
        transform: Object to be moved to the specified transform.
        target_transform: Name of the transform to match to.
    """
    # Get the world-space translation of the target object.
    target_pos: tuple[float, float, float] = cmds.xform(  # type: ignore
        target_transform, query=True, worldSpace=True, translation=True
    )

    # Set the world-space translation of the source object to the target's position.
    cmds.xform(transform, worldSpace=True, translation=target_pos)


def zero_transform(transform: str, local: bool = True) -> None:
    if local:
        set_local_matrix(transform, MMatrix.kIdentity)
    else:
        set_world_matrix(transform, MMatrix.kIdentity)


def zero_rotate_axis(transform: str) -> None:
    node_type = cmds.nodeType(transform)
    if node_type == "joint":
        cmds.joint(transform, edit=True, zeroScaleOrient=True)
    else:
        temp_transform = cmds.group(empty=True, name=f"{transform}_temp")
        match_transform(temp_transform, transform)
        cmds.setAttr(f"{transform}.rotateAxis", 0, 0, 0, type="float3")  # type: ignore
        match_transform(transform, temp_transform)
        cmds.delete(temp_transform)


def clean_parent(transform: str, parent: str, joint_orient: bool = True) -> None:
    """
    Parent a node while preserving its world transform without creating
    Maya's intermediate "compensation" transforms.

    - For transforms: world matrix is preserved.
    - For joints (if joint_orient=True): rotation is baked into jointOrient
      and rotate is zeroed, keeping the joint clean for IK/FK.

    Args:
        transform: Node to reparent.
        parent: New parent node.
        joint_orient: If True, bake rotation into jointOrient for joints.
    """
    object_world_matrix: MMatrix = get_world_matrix(transform)
    node_type = cmds.nodeType(transform)
    cmds.parent(transform, parent, relative=True)

    if node_type == "joint" and joint_orient:
        cmds.setAttr(f"{transform}.jointOrient", 0, 0, 0)  # type: ignore
        set_world_matrix(transform, object_world_matrix)
        # Get current rotation info
        rotate_order = cmds.getAttr(f"{transform}.rotateOrder")
        rotation = cmds.getAttr(f"{transform}.rotate")[0]
        # Convert rotation XYZ rotate order for replacing the joint orient
        euler: MEulerRotation = MEulerRotation(
            MAngle(rotation[0], MAngle.kDegrees).asRadians(),
            MAngle(rotation[1], MAngle.kDegrees).asRadians(),
            MAngle(rotation[2], MAngle.kDegrees).asRadians(),
            rotate_order,
        )
        euler.reorderIt(MEulerRotation.kXYZ)
        # Apply to jointOrient (convert back to degrees)
        cmds.setAttr(
            f"{transform}.jointOrient",
            MAngle(euler.x).asDegrees(),
            MAngle(euler.y).asDegrees(),
            MAngle(euler.z).asDegrees(),
        )
        # Zero the rotate channel
        cmds.setAttr(f"{transform}.rotate", 0, 0, 0)  # type: ignore
    else:
        set_world_matrix(transform, object_world_matrix)
