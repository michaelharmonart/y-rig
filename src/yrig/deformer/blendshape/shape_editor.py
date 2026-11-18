"""Read target selections from Maya's Shape Editor."""

from dataclasses import dataclass

from maya import mel

from yrig.deformer.blendshape.target import get_target_name


@dataclass
class BlendshapeTarget:
    blendshape: str
    target_index: int

    def __str__(self):
        target_name = get_target_name(self.blendshape, self.target_index)
        weight_attr = f"weight[{self.target_index}]"
        return f"{self.blendshape}.{target_name or weight_attr}"


def get_selected_blendshape_targets() -> list[BlendshapeTarget]:
    targets_strings: list[str] = list(mel.eval("getShapeEditorTreeviewSelection(4)") or [])
    targets: list[BlendshapeTarget] = []
    for target_string in targets_strings:
        parts = target_string.split(".")
        blendshape = parts[0]
        target_index = int(parts[1])
        targets.append(BlendshapeTarget(blendshape, target_index))
    return targets
