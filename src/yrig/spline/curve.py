from itertools import chain
from typing import Sequence

from maya import cmds

from yrig.spline import generate_knots
from yrig.transform import matrix_constraint


def bound_curve_from_transforms(
    transforms: Sequence[str],
    name: str,
    parent: str | None = None,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    periodic: bool = False,
    create_pins: bool = True,
    hide: bool = False,
) -> str:
    """
    Create a NURBS curve whose CVs are driven by the given transforms.

    CV control points are connected to each transform's ``.translate`` so the
    curve follows them in real time. When *create_pins* is True, intermediate
    pin nodes are created under a ``{curve_name}_grp`` group and
    translation-constrained to the source transforms; otherwise the CVs connect
    to the source transforms directly.

    Args:
        transforms: Ordered transform names, one per CV.
        name: Name for the created curve transform.
        parent: Optional parent for the curve and its pin group.
        degree: Curve degree (default ``3``, cubic).
        knots: Custom knot vector. Auto-generated when ``None``. First and last
            values are stripped before passing to Maya.
        periodic: Create a closed loop by wrapping the first *degree* CVs.
        create_pins: Create intermediate pin transforms instead of connecting
            CVs to the source transforms directly.

    Returns:
        The created curve transform node name.
    """
    curve_transform_name = name
    curve_group: str | None
    if create_pins:
        if parent:
            curve_group = cmds.group(empty=True, name=f"{curve_transform_name}_grp", parent=parent)
        else:
            curve_group = cmds.group(empty=True, name=f"{curve_transform_name}_grp", world=True)
    else:
        curve_group = None

    full_knots = (
        knots
        if knots is not None
        else generate_knots(len(transforms), degree=degree, periodic=periodic)
    )
    maya_knots: Sequence[float] = full_knots[1:-1]
    extended_cvs = chain(transforms, transforms[:degree]) if periodic else transforms
    curve_transform: str = cmds.curve(
        name=curve_transform_name,
        point=[  # type: ignore
            cmds.xform(cv, query=True, worldSpace=True, translation=True) for cv in extended_cvs
        ],
        periodic=periodic,
        knot=list(maya_knots),
        degree=degree,
    )

    if curve_group is not None:
        cmds.parent(curve_transform, curve_group, relative=True)

    if hide:
        if curve_group is not None:
            cmds.hide(curve_group)
        else:
            cmds.hide(curve_transform)

    extended_cv_mapping: dict[str, str] = {}
    if create_pins:
        for index, transform in enumerate(transforms):
            if curve_group is not None:
                cv = cmds.group(empty=True, name=f"{curve_transform}_cv{index}", parent=curve_group)
            else:
                cv = cmds.group(empty=True, name=f"{curve_transform}_cv{index}", world=True)
            matrix_constraint(
                transform, cv, rotate=False, scale=False, shear=False, keep_offset=False
            )
            if transform not in extended_cv_mapping:
                extended_cv_mapping[transform] = cv
    else:
        extended_cv_mapping = {transform: transform for transform in extended_cvs}

    for index, transform in enumerate(extended_cvs):
        cmds.connectAttr(
            f"{extended_cv_mapping[transform]}.translate",
            f"{curve_transform}.controlPoints[{index}]",
        )
    return curve_transform
