from __future__ import annotations

from dataclasses import dataclass


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
    items: dict[int, BlendShapeTargetItemData]


@dataclass
class BlendShapeTargetItemData:
    components: list[int]
    points: list[tuple[float, float, float]]
