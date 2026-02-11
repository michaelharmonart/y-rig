from typing import Sequence

from maya import cmds

from yrig.name import get_short_name
from yrig.spline.math import generate_knots
from yrig.spline.matrix_spline.core import MatrixSpline
from yrig.spline.matrix_spline.pin import pin_transforms_to_matrix_spline
from yrig.transform.matrix import matrix_constraint


def matrix_spline_from_transforms(
    name: str,
    cv_transforms: Sequence[str],
    pinned_transforms: Sequence[str] | int | None = None,
    parent: str | None = None,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    periodic: bool = False,
    padded: bool = True,
    stretch: bool = True,
    arc_length: bool = True,
    primary_axis: tuple[int, int, int] | None = (0, 1, 0),
    secondary_axis: tuple[int, int, int] | None = (0, 0, 1),
    twist: bool = True,
    align_tangent: bool = True,
) -> MatrixSpline:
    """
    Takes a set of transforms (cvs) and creates a matrix spline and optionally pins transforms to them.
    Args:
        matrix_spline: The matrix spline defention that will drive the pinned transforms.
        pinned_transforms: These transforms will be constrained to the spline.
            If the input is an integer, that many pins will be created and bound to the spline.
        padded: When True, segments are sampled such that the end points have half a segment of spacing from the ends of the spline.
        stretch: Whether to apply automatic scaling along the spline tangent.
        arc_length: When True, the parameters for the spline will be even according to arc length.
        primary_axis (tuple[int, int, int], optional): Local axis of the pinned
            transform that should aim down the spline tangent. Must be one of
            the cardinal axes (±X, ±Y, ±Z). Defaults to (0, 1, 0) (the +Y axis).
        secondary_axis (tuple[int, int, int], optional): Local axis of the pinned
            transform that should be aligned to a secondary reference direction
            from the spline. Used to resolve orientation. Must be one of the
            cardinal axes (±X, ±Y, ±Z) and orthogonal to ``primary_axis``.
            Defaults to (0, 0, 1) (the +Z axis).
        twist (bool): When True the twist is calculated by averaging the secondary axis vector
            as the up vector for the aim matrix. If False no vector is set and the orientation is the swing
            part of a swing twist decomposition.
        align_tangent: When True the pinned segments will align their primary axis along the spline.
    Returns:
        matrix_spline: The matrix spline.
    """
    spline_group: str
    if parent:
        spline_group = cmds.group(empty=True, name=name, parent=parent)
    else:
        spline_group = cmds.group(empty=True, name=name, world=True)
    spline_knots = (
        knots
        if knots is not None
        else generate_knots(len(cv_transforms), degree=degree, periodic=False)
    )

    cv_pins: list[str] = []
    for index, transform in enumerate(cv_transforms):
        cv_pin = cmds.group(empty=True, name=f"{spline_group}_cv{index}", parent=spline_group)
        matrix_constraint(transform, cv_pin, keep_offset=False)
        cv_pins.append(cv_pin)
    matrix_spline = MatrixSpline(
        name=spline_group,
        cv_transforms=cv_pins,
        degree=degree,
        knots=spline_knots,
        periodic=periodic,
    )

    if pinned_transforms is None:
        return matrix_spline

    pins: list[str] = []
    if isinstance(pinned_transforms, int):
        for i in range(pinned_transforms):
            pin_name = f"{matrix_spline.name}_pin{i}"
            pin = cmds.group(empty=True, name=pin_name, parent=spline_group)
            pins.append(pin)
    else:
        for pinned_transform in pinned_transforms:
            pin_name = f"{get_short_name(pinned_transform)}_pin"
            pin = cmds.group(empty=True, name=pin_name, parent=spline_group)
            matrix_constraint(pin, pinned_transform, keep_offset=False)
            pins.append(pin)
    pin_transforms_to_matrix_spline(
        matrix_spline=matrix_spline,
        pinned_transforms=pins,
        padded=padded,
        stretch=stretch,
        arc_length=arc_length,
        primary_axis=primary_axis,
        secondary_axis=secondary_axis,
        twist=twist,
        align_tangent=align_tangent,
    )
    return matrix_spline
