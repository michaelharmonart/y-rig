from __future__ import annotations

from dataclasses import dataclass

from yrig.maya_api.enum import BlendShapePostDeformationOrder


@dataclass
class BlendShapeData:
    name: str
    inputs: dict[int, BlendShapeInputData]
    directory: dict[int, BlendShapeTargetDirectory]
    targets: dict[int, BlendShapeInputTargetData]


@dataclass
class BlendShapeInputData:
    geometry: str
    original_geometry: str | None
    group_id: int = 0
    component_tag_expression: str = "*"


@dataclass
class BlendShapeTargetDirectory:
    name: str
    parent_index: int
    child_indices: list[int]


@dataclass
class BlendShapeInputTargetData:
    groups: dict[int, BlendShapeTargetGroupData]


@dataclass
class BlendShapeTargetGroupData:
    name: str
    mode: BlendShapePostDeformationOrder
    items: dict[int, BlendShapeTargetItemData]


@dataclass
class BlendShapeTargetItemData:
    components: list[int]
    points: list[tuple[float, float, float]]


def get_target_group_indices_map(data: BlendShapeData) -> dict[str, int]:
    """Returns a mapping of target group names to their group indices."""
    name_to_index = {}
    for input_target in data.targets.values():
        for group_index, group_data in input_target.groups.items():
            name_to_index[group_data.name] = group_index
    return name_to_index
