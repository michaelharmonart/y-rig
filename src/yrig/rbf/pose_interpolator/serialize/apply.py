import logging
import math
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from enum import StrEnum

from maya import cmds

from yrig.maya_api.attribute import Attribute
from yrig.maya_api.node import PoseInterpolatorManagerNode, PoseInterpolatorNode
from yrig.name import get_short_name
from yrig.rbf.pose_interpolator.core import resolve_pose_interpolator_shape
from yrig.rbf.pose_interpolator.serialize.directory import (
    filter_pose_interpolator_directory_data,
    find_child_directory_index,
    get_directory_indices,
)
from yrig.transform import create_transform, get_transform

from .data import (
    PoseInterpolatorData,
    PoseInterpolatorDirectoryData,
)

log = logging.getLogger(__name__)


class ExistingPolicy(StrEnum):
    UPDATE = "update"
    SKIP = "skip"
    DUPLICATE = "duplicate"


IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)
IDENTITY_TRANSLATION = (0.0, 0.0, 0.0)


@contextmanager
def preserve_all_pose_interpolator_directories(
    manager: PoseInterpolatorManagerNode,
) -> Generator[None, None, None]:
    """
    When initializing a poseInterpolator plug Maya just decides a directory to add it to.
    This screws up our manual handling of directories on import.

    This might be overkill, but since I don't have the Maya source code on hand
    this is the only way to guarantee that Maya won't screw up directory data.
    """
    directory_indices = manager.pose_interpolator_directory.get_indices()
    snapshot = {
        index: manager.pose_interpolator_directory[index].child_indices.get()
        for index in directory_indices
    }
    try:
        yield
    finally:
        for index, child_indices in snapshot.items():
            manager.pose_interpolator_directory[index].child_indices.set(child_indices)


def _apply_attribute(
    attribute: Attribute,
    source: Attribute | str,
    value: object,
) -> None:
    """Connect an attribute when possible, otherwise set its value."""
    try:
        attribute.connect_from(source)
    except Exception:
        attribute.set(value)


def apply_pose_interpolator_data(
    pose_interpolator: str | PoseInterpolatorNode,
    data: PoseInterpolatorData,
) -> None:
    node = (
        pose_interpolator
        if isinstance(pose_interpolator, PoseInterpolatorNode)
        else PoseInterpolatorNode.from_existing(pose_interpolator)
    )

    node.allow_negative_weights.set(data.allow_negative_weights)
    node.enable_rotation.set(data.enable_rotation)
    node.enable_translation.set(data.enable_translation)
    node.interpolation.set(data.interpolation)
    node.output_smoothing.set(data.output_smoothing)
    node.regularization.set(data.regularization)

    num_drivers = len(data.drivers)

    node.driver.clear()
    for index, driver_data in enumerate(data.drivers):
        driver = node.driver[index]

        driver_node = driver_data.matrix.split(".", 1)[0]
        try:
            driver.driver_matrix.connect_from(driver_data.matrix)
        except Exception:
            log.warning(
                "Couldn't connect %s to from the source attribute specified in the data: %s",
                driver.driver_matrix,
                driver_data.matrix,
            )
        _apply_attribute(driver.driver_orient, f"{driver_node}.jointOrient", driver_data.orient)
        _apply_attribute(
            driver.driver_rotate_axis, f"{driver_node}.rotateAxis", driver_data.rotate_axis
        )
        driver.driver_twist_axis.set(driver_data.twist_axis)
        _apply_attribute(
            driver.driver_rotate_order, f"{driver_node}.rotateOrder", driver_data.rotate_order
        )
        driver.driver_euler_twist.set(driver_data.euler_twist)

        for controller_index, value in driver_data.controllers.items():
            driver.driver_controller[controller_index].set(value)

    node.pose.clear()
    for index, pose_data in enumerate(data.poses):
        pose = node.pose[index]
        pose.pose_name.set(pose_data.name)
        if pose_data.rotations:
            for index, rotation in enumerate(pose_data.rotations):
                pose.pose_rotation[index].set(rotation)
        else:
            for index in range(num_drivers):
                pose.pose_rotation[index].set((0, 0, 0, 1))
        if pose_data.translations:
            for index, translation in enumerate(pose_data.translations):
                pose.pose_translation[index].set(translation)
        else:
            for index in range(num_drivers):
                pose.pose_translation[index].set((0, 0, 0))
        pose.is_independent.set(pose_data.independent)
        pose.pose_rotation_falloff.set(pose_data.rotation_falloff)
        if not math.isclose(pose_data.translation_falloff, 0):
            pose.pose_translation_falloff.set(pose_data.translation_falloff)
        pose.pose_type.set(pose_data.pose_type)
        pose.pose_falloff.set(pose_data.gaussian_falloff)
        pose.is_enabled.set(pose_data.is_enabled)

        for controller_index, controller_data in pose_data.controller_data.items():
            controller = pose.pose_controller_data[controller_index]

            for item_index, item_data in controller_data.items.items():
                item = controller.pose_controller_data[item_index]
                item.pose_controller_data_item_name.set(item_data.name)
                item.pose_controller_data_item_type.set(item_data.type)
                item.pose_controller_data_item_value.set(item_data.value)

    for index, output in enumerate(data.outputs):
        output_attr = node.output[index]
        desired_connections = set(output)
        current_connections = {str(attr) for attr in output_attr.get_outputs()}
        for destination in current_connections - desired_connections:
            output_attr.disconnect_from(destination)

        for destination in desired_connections - current_connections:
            try:
                output_attr.connect_to(destination)
            except Exception:
                log.warning(
                    "Couldn't connect %s to the destination specified in the data: %s",
                    output_attr,
                    destination,
                )


def add_pose_interpolator(
    manager: PoseInterpolatorManagerNode,
    data: PoseInterpolatorData,
    parent: str | None = None,
    parent_directory_index: int = 0,
) -> tuple[str, PoseInterpolatorNode]:
    """Add poseInterpolator to manager and apply data. Returns the transform and shape node of the created poseInterpolator."""
    pose_interpolator = PoseInterpolatorNode().create(name=data.name)
    pose_interpolator_transform = get_transform(str(pose_interpolator))
    pose_interpolator_transform = cmds.rename(
        pose_interpolator_transform, data.name.removesuffix("Shape"), ignoreShape=True
    )
    if parent is not None:
        cmds.parent(pose_interpolator_transform, parent)
    target_index = manager.pose_interpolator_parent.next_available_index()
    parent_directory = manager.pose_interpolator_directory[parent_directory_index]
    with preserve_all_pose_interpolator_directories(manager):
        pose_interpolator.mid_layer_id.set(target_index)
        pose_interpolator.mid_layer_parent.connect_to(
            manager.pose_interpolator_parent[target_index]
        )
    original_child_indices = parent_directory.child_indices.get()
    apply_pose_interpolator_data(pose_interpolator, data)
    parent_directory.child_indices.set([*original_child_indices, target_index])
    return pose_interpolator_transform, pose_interpolator


def find_child_pose_interpolator(
    manager: PoseInterpolatorManagerNode,
    parent_directory_index: int,
    name: str,
) -> PoseInterpolatorNode | None:
    """Return the poseInterpolator called ``name`` directly inside the directory, or None."""
    child_indices = manager.pose_interpolator_directory[parent_directory_index].child_indices.get()
    for child_index in child_indices:
        if child_index < 0:  # negative entries are directories
            continue
        parent_attr = manager.pose_interpolator_parent[child_index]
        source_attr = parent_attr.get_input()
        if source_attr is None:
            raise RuntimeError(f"Couldn't find a poseInterpolator connected to {source_attr}")
        source_node = str(source_attr).split(".", 1)[0]
        pose_interpolator_shape = resolve_pose_interpolator_shape(source_node)
        pose_interpolator_transform = get_transform(pose_interpolator_shape)
        if name in {
            get_short_name(pose_interpolator_shape),
            get_short_name(pose_interpolator_transform),
        }:
            source_node = str(source_attr).split(".", 1)[0]
            resolve_pose_interpolator_shape(source_node)
            return PoseInterpolatorNode.from_existing(source_node)
    return None


def merge_pose_interpolator(
    manager: PoseInterpolatorManagerNode,
    data: PoseInterpolatorData,
    parent: str | None,
    parent_directory_index: int,
    existing_policy: ExistingPolicy = ExistingPolicy.UPDATE,
) -> tuple[str, PoseInterpolatorNode] | None:
    """
    Add the poseInterpolator, or reuse an existing one with the same name in the directory.
    Returns None if it already exists and the policy is "skip".
    """
    if existing_policy != "duplicate":
        existing = find_child_pose_interpolator(manager, parent_directory_index, data.name)
        if existing is not None:
            if existing_policy == "skip":
                return None
            apply_pose_interpolator_data(existing, data)
            transform = get_transform(str(existing))
            if parent is not None:
                current_parent = cmds.listRelatives(transform, parent=True, fullPath=True)
                if not current_parent or current_parent[0] != cmds.ls(parent, long=True)[0]:
                    transform = cmds.parent(transform, parent)[0]
            return transform, existing

    return add_pose_interpolator(manager, data, parent, parent_directory_index)


def add_pose_interpolator_directory(
    manager: PoseInterpolatorManagerNode,
    data: PoseInterpolatorDirectoryData,
    parent_directory_index: int,
) -> int:
    """Add poseInterpolator directory to manager and apply data. Returns the index at which the poseInterpolator directory was added."""
    directory_index = manager.pose_interpolator_directory.next_available_index()
    directory = manager.pose_interpolator_directory[directory_index]

    directory.directory_name.set(data.name)
    directory.parent_index.set(parent_directory_index)
    parent_directory_child_indices = manager.pose_interpolator_directory[
        parent_directory_index
    ].child_indices.get()

    new_child_indices = [*parent_directory_child_indices, -directory_index]
    manager.pose_interpolator_directory[parent_directory_index].child_indices.set(new_child_indices)
    return directory_index


def _get_or_create_folder(name: str, parent: str | None) -> str:
    named = f"{name}_rbf"
    path = f"{parent}|{named}" if parent else f"|{named}"
    if cmds.objExists(path):
        return cmds.ls(path, long=True)[0]
    return create_transform(named, parent=parent)


def _import_directory(
    manager: PoseInterpolatorManagerNode,
    data: PoseInterpolatorDirectoryData,
    parent_directory_index: int,
    parent: str | None,
    create_folders: bool = False,
    merge_directories: bool = True,
    existing_policy: ExistingPolicy = ExistingPolicy.UPDATE,
) -> list[tuple[str, PoseInterpolatorNode]]:
    """Create a directory (and everything below it) under parent_directory_index."""
    directory_index = None
    if merge_directories:
        directory_index = find_child_directory_index(manager, parent_directory_index, data.name)
    if directory_index is None:
        directory_index = add_pose_interpolator_directory(manager, data, parent_directory_index)

    if create_folders:
        child_parent = (
            _get_or_create_folder(data.name, parent)
            if merge_directories
            else create_transform(data.name, parent=parent)
        )
    else:
        child_parent = parent

    return _import_directory_contents(
        manager,
        data,
        directory_index,
        parent=child_parent,
        existing_policy=existing_policy,
        create_folders=create_folders,
    )


def _import_directory_contents(
    manager: PoseInterpolatorManagerNode,
    data: PoseInterpolatorDirectoryData,
    parent_directory_index: int,
    parent: str | None,
    create_folders: bool = False,
    merge_directories: bool = True,
    existing_policy: ExistingPolicy = ExistingPolicy.UPDATE,
) -> list[tuple[str, PoseInterpolatorNode]]:
    created = []
    for child in data.directories:
        created.extend(
            _import_directory(
                manager,
                child,
                parent_directory_index,
                parent,
                create_folders,
                merge_directories,
                existing_policy,
            )
        )
    for pose_interpolator_data in data.pose_interpolators:
        result = merge_pose_interpolator(
            manager, pose_interpolator_data, parent, parent_directory_index, existing_policy
        )
        if result is not None:
            created.append(result)
    return created


def apply_pose_interpolator_directory_data(
    data: PoseInterpolatorDirectoryData,
    *,
    directories: Iterable[str] | None = None,
    pose_interpolators: Iterable[str] | None = None,
    parent_directory: str | None = None,
    parent: str | None = None,
    create_folders: bool = False,
    merge_directories: bool = True,
    existing_policy: ExistingPolicy = ExistingPolicy.UPDATE,
) -> list[tuple[str, PoseInterpolatorNode]]:
    """
    Import directory data into the poseInterpolatorManager.

    Args:
        data: Root directory data loaded from a file.
        directories: Directory names to import (with everything below them).
        pose_interpolators: poseInterpolator names (shape or transform) to import.
        parent_directory: Manager directory to import into. Defaults to the root.
        parent: Optional DAG parent for the created poseInterpolator transforms.

    If neither ``directories`` nor ``pose_interpolators`` is given, everything in
    the file is imported.

    Returns:
        The (transform, shape node) pair of every created poseInterpolator.
    """
    manager = PoseInterpolatorManagerNode.from_existing("poseInterpolatorManager")

    parent_directory_index = 0
    if parent_directory is not None:
        parent_directory_index = get_directory_indices(manager, {parent_directory})[
            parent_directory
        ]

    directory_names = set(directories) if directories is not None else None
    pose_interpolator_names = set(pose_interpolators) if pose_interpolators is not None else None

    filtered_data = filter_pose_interpolator_directory_data(
        data, directory_names, pose_interpolator_names
    )
    created = _import_directory_contents(
        manager,
        filtered_data,
        parent_directory_index,
        parent,
        create_folders,
        merge_directories,
        existing_policy,
    )
    return created
