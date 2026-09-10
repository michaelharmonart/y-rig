from collections.abc import Collection
from dataclasses import replace

from .data import BlendShapeTargetDirectory


def compute_needed_indices(
    directory_data: dict[int, BlendShapeTargetDirectory],
    directories_to_keep: Collection[str],
    group_indices_to_keep: Collection[int],
) -> set[int]:
    """
    Walk the directory tree and return the set of directory indices
    (negative) and group indices (positive) that are needed given the
    requested directory names and/or explicit group indices.
    """
    needed_indices: set[int] = set()

    def mark_children(index: int) -> None:
        if index in needed_indices:
            return
        needed_indices.add(index)
        if index < 0:
            directory = directory_data[-index]
            for child in directory.child_indices:
                mark_children(child)

    def mark_needed(index: int) -> bool:
        if index >= 0:
            return index in group_indices_to_keep
        directory = directory_data[-index]
        if directory.name in directories_to_keep:
            mark_children(index)
            return True
        if any(mark_needed(child) for child in directory.child_indices):
            needed_indices.add(index)
            return True
        return False

    for index in directory_data:
        if index != 0:
            mark_needed(-index)

    return needed_indices


def prune_blendshape_directory_dict(
    directory_data: dict[int, BlendShapeTargetDirectory],
    directories_to_keep: Collection[str],
    group_indices_to_keep: Collection[int],
) -> dict[int, BlendShapeTargetDirectory]:
    needed_indices = compute_needed_indices(
        directory_data, directories_to_keep, group_indices_to_keep
    )

    pruned_directory_data = {
        index: replace(
            data,
            child_indices=[
                child
                for child in data.child_indices
                if child in needed_indices or child in group_indices_to_keep
            ],
        )
        for index, data in directory_data.items()
        if index == 0 or -index in needed_indices or index in group_indices_to_keep
    }

    return pruned_directory_data


def resolve_needed_group_indices(
    directory_data: dict[int, BlendShapeTargetDirectory],
    directories_to_keep: Collection[str] | None = None,
    group_indices_to_keep: Collection[int] | None = None,
) -> set[int] | None:
    """
    Resolve the final set of blendShape group (weight) indices to import/export,
    given directory-name and/or explicit-target filters.

    Returns ``None`` if no filtering was requested, meaning all groups
    are needed.
    """
    if directories_to_keep is None and group_indices_to_keep is None:
        return None
    needed_indices = compute_needed_indices(
        directory_data,
        directories_to_keep=directories_to_keep or set(),
        group_indices_to_keep=group_indices_to_keep or set(),
    )
    resolved = {index for index in needed_indices if index >= 0}
    resolved.update(group_indices_to_keep or set())
    return resolved
