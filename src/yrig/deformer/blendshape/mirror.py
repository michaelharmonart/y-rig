from collections.abc import Callable, Iterable

from maya import cmds

from yrig.deformer.blendshape.core import (
    copy_target_deltas,
    get_blendshape_target_names,
    get_name_to_target_index_map,
)
from yrig.maya_api.node import BlendShapeNode
from yrig.name import Side, filter_side, flip_side


def mirror_blendshape_targets(
    blendshape: BlendShapeNode | str,
    targets: Iterable[str] | str,
    *,
    map_mirror_target: Callable[[str], str] | None = None,
) -> dict[str, str]:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    original_targets = [targets] if isinstance(targets, str) else list(targets)
    mirrored_targets = [
        map_mirror_target(target) if map_mirror_target is not None else flip_side(target)
        for target in original_targets
    ]
    name_to_index_map = get_name_to_target_index_map(str(blendshape_node))
    mirrored: dict[str, str] = {}
    for original_target, mirrored_target in zip(original_targets, mirrored_targets, strict=True):
        original_target_index = name_to_index_map[original_target]
        if mirrored_target not in name_to_index_map:
            raise NotImplementedError(
                "This should create a mirror target if it doesn't already exist. Ideally it would even put it in the right directory."
            )
        mirrored_target_index = name_to_index_map[mirrored_target]
        for input_index in blendshape_node.input_target.get_indices():
            input_target = blendshape_node.input_target[input_index]
            existing_target_group_indices = set(input_target.input_target_group.get_indices())
            if original_target_index not in existing_target_group_indices:
                continue

            # TODO: Switch this out for our own shape mirroring logic. This is buns slow.
            copy_target_deltas(
                blendshape_node, input_index, original_target_index, mirrored_target_index
            )
            cmds.blendShape(
                str(blendshape_node),
                edit=True,
                flipTarget=(input_index, mirrored_target_index),
                symmetryAxis="X",
                symmetrySpace=1,  # Object space
            )

            mirrored[original_target] = mirrored_target
    return mirrored


def mirror_all_blendshape_targets(
    blendshape: BlendShapeNode | str,
    *,
    left_to_right: bool = True,
    map_mirror_target: Callable[[str], str] | None = None,
) -> dict[str, str]:
    blendshape_node = (
        blendshape
        if isinstance(blendshape, BlendShapeNode)
        else BlendShapeNode.from_existing(blendshape)
    )
    source_side = Side.LEFT if left_to_right else Side.RIGHT

    targets = filter_side(get_blendshape_target_names(str(blendshape_node)), source_side)

    return mirror_blendshape_targets(
        blendshape_node,
        targets,
        map_mirror_target=map_mirror_target,
    )
