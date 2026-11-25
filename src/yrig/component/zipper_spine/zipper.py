from maya import cmds
from maya.api.OpenMaya import MMatrix

from yrig.control import create_control

from .guide.core import GuideInfo
from .utils import get_closest_curve_matrix


def build_zipper(
    guide_curve: GuideInfo,
    spine_drivers: list[str],
    parent: str,
    size: float = 6.0,
) -> None:

    curve = guide_curve.name

    for i, transform in enumerate(spine_drivers):
        matrix = get_closest_curve_matrix(
            curve=curve,
            transform=transform,
        )

        ctrl = create_control(
            name=f"zipper_{i + 2:02d}_M",
            parent=parent,
            transform=MMatrix(matrix),
            size=size,
            control_shape="sphere",
        )

        cmds.parentConstraint(transform, ctrl.offset, maintainOffset=True)
