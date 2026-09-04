from collections.abc import Callable, Sequence

from maya import cmds
from maya.api.OpenMaya import MFnNurbsCurve, MPoint, MSpace

from yrig.maya_api.enum import Axis
from yrig.maya_api.node import MotionPathNode, MultiplyPointByMatrixNode
from yrig.maya_api.utils import get_dag_path
from yrig.name import get_short_name
from yrig.shape import get_shape
from yrig.spline import generate_knots
from yrig.spline.math import collapse_periodic_cv_list, create_periodic_cv_list
from yrig.transform import create_transform, get_shapes
from yrig.transform.matrix import localize_world_matrix
from yrig.transform.utils import set_position


def bound_curve_from_transforms(
    transforms: Sequence[str],
    name: str,
    parent: str | None = None,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    periodic: bool = False,
    hide: bool = False,
) -> str:
    """
    Create a NURBS curve whose CVs are driven by the given transforms.

    Args:
        transforms: Ordered transform names, one per CV.
        name: Name for the created curve transform.
        parent: Optional parent for the curve.
        degree: Curve degree (default ``3``, cubic).
        knots: Custom knot vector. Auto-generated when ``None``. First and last
            values are stripped before passing to Maya.
        periodic: Create a closed loop by wrapping the first *degree* CVs.

    Returns:
        The name of the created curve transform node.
    """
    curve_transform_name = name
    full_knots = (
        knots
        if knots is not None
        else generate_knots(len(transforms), degree=degree, periodic=periodic)
    )
    maya_knots: Sequence[float] = full_knots[1:-1]
    extended_cvs = create_periodic_cv_list(transforms, degree) if periodic else transforms
    cv_positions: list[tuple[float, float, float]] = [  # type: ignore
        cmds.xform(cv, query=True, worldSpace=True, translation=True) for cv in extended_cvs
    ]
    curve_transform: str = cmds.curve(
        name=curve_transform_name,
        point=cv_positions,
        periodic=periodic,
        knot=list(maya_knots),
        degree=degree,
    )
    if parent is not None:
        cmds.parent(curve_transform, parent, relative=True)
    curve_shape = get_shapes(curve_transform)[0]
    cmds.rename(curve_shape, f"{curve_transform_name}Shape")
    if hide:
        cmds.hide(curve_transform)

    for index, transform in enumerate(extended_cvs):
        localize = localize_world_matrix(transform, curve_transform)
        translation = MultiplyPointByMatrixNode.create(f"{curve_transform}_cv{index}_position")
        translation.input_matrix.connect_from(localize.matrix_sum)
        translation.output.connect_to(f"{curve_transform}.controlPoints[{index}]")
    return curve_transform


def get_curve_cvs(curve: str, world_space: bool = True) -> list[MPoint]:
    """
    Get all CV positions from a NURBS curve.

    Args:
        curve: Curve transform or nurbsCurve shape node.
        world_space: Whether to return CV positions in world space.
            If ``False``, positions are returned in object space.

    Returns:
        List of CV positions as ``MPoint`` objects.
    """
    shape = get_shape(curve)
    if shape is None:
        raise RuntimeError(f"{curve} had no shape node!")
    if not cmds.nodeType(shape) == "nurbsCurve":
        raise RuntimeError(f"{curve} is not a nurbsCurve")
    dag_path = get_dag_path(shape)
    curve_fn = MFnNurbsCurve(dag_path)
    return [
        MPoint(cv) for cv in curve_fn.cvPositions(MSpace.kWorld if world_space else MSpace.kObject)
    ]


def create_transforms_at_curve_cvs(
    curve: str,
    name_format: str | Callable[[int], str] | None = None,
    parent: str | None = None,
    create_periodic_duplicate_cvs: bool = False,
) -> list[str]:
    """
    Create transforms positioned at each CV of a NURBS curve.

    Args:
        curve: Curve transform or nurbsCurve shape node.
        name_format: Naming rule for created transforms.
            - None: names are generated as "{curve_name}_cv{index}"
            - str: names are generated as "{name_format}{index}"
            - callable: called with the CV index and must return the name
        parent: Optional parent transform for created nodes.

    Returns:
        List of created transform node names.
    """
    curve_cvs = get_curve_cvs(curve)

    shape = get_shape(curve)
    if shape is None:
        raise RuntimeError(f"{curve} had no shape node!")

    if not create_periodic_duplicate_cvs and cmds.getAttr(f"{shape}.form") == 2:
        # Remove duplicated CVs from periodic curves
        degree = cmds.getAttr(f"{shape}.degree")
        curve_cvs = collapse_periodic_cv_list(curve_cvs, degree)

    transforms: list[str] = []
    for index, cv in enumerate(curve_cvs):
        transform_name: str
        if name_format is None:
            curve_name = get_short_name(curve)
            transform_name = f"{curve_name}_cv{index}"
        elif isinstance(name_format, str):
            transform_name = f"{name_format}{index}"
        else:
            transform_name = name_format(index)
        transform = create_transform(transform_name, parent)
        set_position(transform, cv)
        transforms.append(transform)
    return transforms


def pin_to_curve_with_motion_path(
    curve: str,
    pinned_transform: str,
    parameter: float,
    arc_length: bool = True,
    orient: bool = True,
    front_axis: Axis = Axis.X,
    up_axis: Axis = Axis.Z,
    up_vector: tuple[float, float, float] = (0, 0, 1),
) -> MotionPathNode:
    curve_shape = get_shape(curve)
    if curve_shape is None:
        raise RuntimeError(f"{curve} had no curve shape")
    motion_path = MotionPathNode.create(f"{pinned_transform}_motion_path")
    motion_path.geometry_path.connect_from(f"{curve_shape}.local")
    motion_path.u_value.set(parameter)
    motion_path.fraction_mode.set(arc_length)
    motion_path.follow.set(orient)
    if orient:
        motion_path.rotate.connect_to(f"{pinned_transform}.rotate")
        motion_path.rotate_order.connect_from(f"{pinned_transform}.rotateOrder")
        motion_path.front_axis.set(front_axis)
        motion_path.up_axis.set(up_axis)
        motion_path.world_up_vector.set(up_vector)
    motion_path.all_coordinates.connect_to(f"{pinned_transform}.translate")

    return motion_path
