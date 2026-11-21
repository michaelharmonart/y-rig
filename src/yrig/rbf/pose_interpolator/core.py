import logging
from collections.abc import Callable, Iterable

from maya import cmds

from yrig.maya_api.attribute import IntegerAttribute
from yrig.maya_api.node import PoseInterpolatorManagerNode, PoseInterpolatorNode
from yrig.transform import create_transform, get_transform

log = logging.getLogger(__name__)


def resolve_pose_interpolator_shape(pose_interpolator: str) -> str:
    if cmds.nodeType(pose_interpolator) == "poseInterpolator":
        return pose_interpolator
    shapes = cmds.listRelatives(
        pose_interpolator,
        shapes=True,
        type="poseInterpolator",
    )
    if not shapes:
        raise RuntimeError(f"Couldn't find a shape for {pose_interpolator}")
    return shapes[0]


def resolve_pose_interpolator_node(
    pose_interpolator: PoseInterpolatorNode | str,
) -> PoseInterpolatorNode:
    pose_interpolator_node = (
        pose_interpolator
        if isinstance(pose_interpolator, PoseInterpolatorNode)
        else PoseInterpolatorNode.from_existing(
            name=resolve_pose_interpolator_shape(pose_interpolator)
        )
    )
    return pose_interpolator_node


def create_pose_interpolator(
    manager: PoseInterpolatorManagerNode,
    shape_name: str,
    parent: str | None = None,
    parent_directory_index: int = 0,
) -> tuple[str, PoseInterpolatorNode]:
    """Add poseInterpolator to manager and apply data. Returns the transform and shape node of the created poseInterpolator."""
    pose_interpolator = PoseInterpolatorNode().create(name=shape_name)
    pose_interpolator_transform = get_transform(str(pose_interpolator))
    pose_interpolator_transform = cmds.rename(
        pose_interpolator_transform, shape_name.removesuffix("Shape"), ignoreShape=True
    )
    if parent is not None:
        cmds.parent(pose_interpolator_transform, parent)
    target_index = manager.pose_interpolator_parent.next_available_index()
    pose_interpolator.mid_layer_id.set(value=target_index)
    pose_interpolator.mid_layer_parent.set(parent_directory_index)
    pose_interpolator.mid_layer_parent.connect_to(manager.pose_interpolator_parent[target_index])
    return pose_interpolator_transform, pose_interpolator


def get_pose_index(pose_interpolator_shape: str, pose_name: str) -> int:
    attr = f"{pose_interpolator_shape}.pose"

    for index in cmds.getAttr(attr, multiIndices=True) or []:
        name = cmds.getAttr(f"{attr}[{index}].poseName")
        if name == pose_name:
            return index

    raise RuntimeError(f"Couldn't resolve and index for the pose {pose_name}")


def _reslove_pose_index(pose_interpolator: str, pose: str | int) -> int:
    shape = resolve_pose_interpolator_shape(pose_interpolator)
    return pose if isinstance(pose, int) else get_pose_index(shape, pose)


def get_pose_interpolator_blendshapes(pose_interpolator: str) -> list[str]:
    shape = resolve_pose_interpolator_shape(pose_interpolator)

    blendshapes: list[str] = []
    for index in cmds.poseInterpolator(shape, query=True, index=True) or []:
        plug = f"{shape}.output[{index}]"

        for node in cmds.listConnections(plug, source=False) or []:
            if cmds.nodeType(node) == "blendShape":
                blendshapes.append(node)

    return list(blendshapes)


# TODO make this keep the ordering of the directory children by using the childIndices attr.
def group_pose_interpolators_by_directory(
    pose_interpolators: Iterable[str],
    parent: str | None = None,
    group_namer: Callable[[str], str] | None = None,
) -> list[str]:
    manager = PoseInterpolatorManagerNode.from_existing("poseInterpolatorManager")
    created_groups = []
    index_group_map: dict[int, str] = {}

    def get_group_name(directory_index: int) -> str:
        directory_name = manager.pose_interpolator_directory[directory_index].directory_name.get()
        return group_namer(directory_name) if group_namer else f"{directory_name}_rbf"

    for pose_interpolator in pose_interpolators:
        shape = resolve_pose_interpolator_shape(pose_interpolator)
        transform: str | None = next(
            iter(cmds.listRelatives(shape, parent=True, type="transform") or []),
            None,
        )
        destinations = (
            cmds.listConnections(
                shape,
                source=False,
                destination=True,
                plugs=True,
            )
            or []
        )
        if not destinations:
            continue

        directory_index: int = IntegerAttribute(destinations[0]).get()

        # Walk upward only until we find an existing group.
        missing: list[int] = []
        group_parent = parent

        while directory_index != 0:
            if directory_index in index_group_map:
                group_parent = index_group_map[directory_index]
                break

            group_name = get_group_name(directory_index)

            if cmds.objExists(group_name):
                index_group_map[directory_index] = group_name
                group_parent = group_name
                break

            missing.append(directory_index)

            directory_index = manager.pose_interpolator_directory[
                directory_index
            ].parent_index.get()

        # Create the missing part of the hierarchy from top -> bottom.
        for directory_index in reversed(missing):
            group_name = get_group_name(directory_index)

            create_transform(group_name, group_parent)

            index_group_map[directory_index] = group_name
            created_groups.append(group_name)

            group_parent = group_name
        if transform and group_parent:
            cmds.parent(transform, group_parent, relative=True)

    return created_groups


def force_pose_interpolator_update(pose_interpolator: str | PoseInterpolatorNode) -> None:
    # This is a hack to get the pose interpolator to update its internal state and use our newly injected data.
    # If you find out a better way to do this, great!
    attr = f"{pose_interpolator}.regularization"
    cmds.setAttr(attr, cmds.getAttr(attr))


def _validate_pose_interpolators(
    pose_interpolators: list[str],
) -> None:
    for pose_interpolator in pose_interpolators:
        if (
            not cmds.objExists(pose_interpolator)
            or cmds.nodeType(pose_interpolator) != "poseInterpolator"
        ):
            raise RuntimeError(f"Not a pose interpolator: {pose_interpolator}")
