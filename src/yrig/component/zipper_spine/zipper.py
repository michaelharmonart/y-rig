
from yrig.control import create_control

from .guide.core import GuideInfo, create_guide_from_position
from .utils import get_closest_curve_matrix


def build_zipper(
    guide_curve: GuideInfo,
    spine_drivers: list[str],
    parent: str,
    size: float = 1.0,
) -> None:
    controls = []

    curve = guide_curve.name

    for i, transform in enumerate(spine_drivers):
        matrix = get_closest_curve_matrix(
            curve=curve,
            transform=transform,
        )

        guide = create_guide_from_position(
            pos=matrix, guide_name=f"zipper_{(i + 2):02d}_M", parent="guides"
        )

        ctrl = create_control(
            name=guide.descriptor,
            parent=parent,
            transform=guide.name,
            size=size,
            control_shape="sphere",
        )

        controls.append(ctrl)
