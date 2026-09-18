from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from maya import cmds
from maya.api.OpenMaya import MMatrix

from yrig.build.mgear_api.joint import add_to_joint_set
from yrig.control.core import Control
from yrig.transform import match_transform, matrix_constraint, set_world_matrix

JOINT_SUFFIX: str = "_jnt"

_joint_collection: ContextVar[list[str] | None] = ContextVar("joint_collection", default=None)


@contextmanager
def collect_joints() -> Iterator[list[str]]:
    """
    Collect joints created inside this block.

    Any joints created within the `with` statement are added to a list,
    which is returned when the block finishes. Nested blocks are supported,
    and inner results are included in the outer list.

    Returns:
        list[str]: Joint names created in the block.
    """
    # Create a bucket to collect the joints created in the with block
    # then put it into the ContextVar so that _register_joint will add to this bucket
    bucket: list[str] = []
    parent_bucket = _joint_collection.get()
    token = _joint_collection.set(bucket)
    try:
        yield bucket
    finally:
        # Restore the previous state
        _joint_collection.reset(token)
        # If there was a parent, bubble up the results
        if parent_bucket is not None:
            parent_bucket.extend(bucket)


def _register_joint(joint: str) -> None:
    bucket = _joint_collection.get()
    if bucket is not None:
        bucket.append(joint)


def create_joint(
    name: str,
    transform: str | Control | MMatrix | None = None,
    parent: str | None = None,
    *,
    connect: bool = True,
    radius: float = 1,
    add_to_set: bool = True,
) -> str:
    joint = cmds.createNode("joint", name=f"{name}{JOINT_SUFFIX}")

    if parent is not None:
        cmds.parent(joint, parent, relative=True)
    source_transform: str | None = None
    if transform is None:
        pass
    elif isinstance(transform, Control):
        source_transform = transform.transform
    elif isinstance(transform, str):
        source_transform = transform
    elif isinstance(transform, MMatrix):
        set_world_matrix(joint, transform, use_joint_orient=True)
    else:
        raise RuntimeError(f"{transform} is not a valid transform name or MMatrix")
    if source_transform is not None:
        match_transform(joint, source_transform, use_joint_orient=True)
        if connect:
            matrix_constraint(source_transform, joint, False, use_joint_orient=True)

    if radius != 1:
        cmds.setAttr(f"{joint}.radius", radius)  # type: ignore

    _register_joint(joint)
    # This is mGear specific and may need changed if you stop using mGear.
    if add_to_set:
        add_to_joint_set(joint)
    return joint
