"""Read target selections from Maya's Shape Editor."""

from dataclasses import dataclass

from maya import mel


@dataclass
class BlendshapeTarget:
    blendshape: str
    target_index: int


def get_selected_blendshape_targets() -> list[BlendshapeTarget]:
    targets_strings: list[str] = list(mel.eval("getShapeEditorTreeviewSelection(4)") or [])
    targets: list[BlendshapeTarget] = []
    for target_string in targets_strings:
        parts = target_string.split(".")
        blendshape = parts[0]
        target_index = int(parts[1])
        targets.append(BlendshapeTarget(blendshape, target_index))
    return targets
