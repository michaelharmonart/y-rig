from collections.abc import Iterable
from itertools import zip_longest

from maya import cmds
from maya.api.OpenMaya import MDagPath, MFnNurbsSurface, MMatrix, MPoint, MSelectionList, MSpace

from yrig.math import remap
from yrig.maya_api.attribute import (
    ClosestPointOnSurfaceResultAttribute,
    MatrixAttribute,
)
from yrig.maya_api.enum import Axis
from yrig.maya_api.node import (
    ClosestPointOnSurfaceNode,
    MultiplyPointByMatrixNode,
    UvPinNode,
)
from yrig.name import get_short_name
from yrig.transform import get_shape
from yrig.transform.matrix import (
    drive_transform_with_matrix,
    get_world_matrix,
    matrix_normal_orient_constraint,
    multiply_matrices,
)
from yrig.transform.utils import get_position


def closest_point_on_surface(
    surface: str, position: MPoint | tuple[float, float, float], world_space: bool = True
) -> tuple[MPoint, tuple[float, float]]:
    """
    Return the closest point and UV on a NURBS surface to the given position.

    Args:
        surface: The NURBS surface transform or shape.
        position: Query point.

    Returns:
        Tuple of (closest point, (u, v)).
    """
    shape = get_shape(surface)
    msel: MSelectionList = MSelectionList()
    msel.add(shape)
    surface_dag: MDagPath = msel.getDagPath(0)
    fn_surface: MFnNurbsSurface = MFnNurbsSurface(surface_dag)

    test_point: MPoint = position if isinstance(position, MPoint) else MPoint(*position)

    result_point, u, v = fn_surface.closestPoint(
        test_point, space=MSpace.kWorld if world_space else MSpace.kObject
    )
    return (result_point, (u, v))


def surface_uv_domain(surface: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return the knot domain of a NURBS surface in U and V.
    (minimum and maximum UV parameter values for the surface)

    Args:
        surface: The NURBS surface transform or shape.

    Returns:
        Tuple of ((u_min, u_max), (v_min, v_max)).
    """
    shape = get_shape(surface)
    msel: MSelectionList = MSelectionList()
    msel.add(shape)
    surface_dag: MDagPath = msel.getDagPath(0)
    fn_surface: MFnNurbsSurface = MFnNurbsSurface(surface_dag)
    return (fn_surface.knotDomainInU, fn_surface.knotDomainInV)


def get_surface_shapes(surface: str) -> tuple[str, str, str]:
    """Return (primary_shape, original_shape, shape_output_attr)."""
    shapes: list[str] = cmds.listRelatives(surface, shapes=True, noIntermediate=True) or []
    if not shapes:
        cmds.error(f"No shape nodes found on surface: {surface}")
    primary_shape: str = shapes[0]
    original_shape_geo: str = cmds.deformableShape(primary_shape, originalGeometry=True)[0]  # type: ignore
    if not original_shape_geo:
        original_shape_geo = cmds.deformableShape(primary_shape, createOriginalGeometry=True)[0]  # type: ignore
    # the return from deformableShape is in the form ["shapeName.local"] so we pull the node name with a split
    original_shape: str = original_shape_geo.split(".", 1)[0]
    shape_output: str = cmds.deformableShape(primary_shape, worldShapeOutAttr=True)[0]  # type: ignore

    return primary_shape, original_shape, shape_output


def _resolve_uv_for_pin(
    surface: str, object_to_pin: str, uv: tuple[float, float] | None = None, normalize: bool = False
) -> tuple[float, float]:
    if uv is not None:
        resolved_uv = uv
    else:
        object_to_pin_position = get_position(object_to_pin)
        _, sampled_uv = closest_point_on_surface(surface, object_to_pin_position)
        if normalize:
            surface_domain_u, surface_domain_v = surface_uv_domain(surface)
            remapped_u = remap(sampled_uv[0], surface_domain_u)
            remapped_v = remap(sampled_uv[1], surface_domain_v)
            resolved_uv = (remapped_u, remapped_v)
        else:
            resolved_uv = sampled_uv

    return resolved_uv


def uv_pin(
    surface: str,
    object_to_pin: str,
    uv: tuple[float, float] | None = None,
    normalize: bool = False,
    normal_axis: Axis = Axis.Z,
    tangent_axis: Axis = Axis.X,
    uv_pin_node: UvPinNode | None = None,
    keep_offset: bool = False,
    drive_rotate: bool = True,
    drive_translate: bool = True,
) -> tuple[UvPinNode, int]:
    """
    Create a uvPin node that pins an object to a given surface at specified UV coordinates.

    Args:
        surface: The name of the surface (mesh or NURBS) to pin to.
        object_to_pin: The name of the object to be pinned.
        uv: The UV coordinate to pin at, if None it will be pinned to the closest point.
        When false, the pinned object has inheritsTransform disabled to prevent double transforms.
        normalize: Enable Isoparm normalization (NURBS UV will be remapped between 0-1).
        normal_axis: Normal axis of the generated uvPin, can be x y z -x -y -z.
        tangent_axis: Tangent axis of the generated uvPin, can be x y z -x -y -z.
        uv_pin_node: When specified the object will be pinned as an additional slot in the given uvPin node.
        keep_offset: When True, the pinned object will be offset to be in
            the same world space placement as before being pinned.
        drive: When True the pinned objects transforms will be driven by the uvPin.
    Returns:
        The created UVPin node.
    """

    primary_shape, original_shape, shape_output = get_surface_shapes(surface)
    pin_name = f"{get_short_name(object_to_pin)}_uvPin"

    if uv_pin_node is None:
        # Create the UVPin node and connect it.
        uv_pin_node = UvPinNode.create(pin_name)
        uv_pin_node.original_geometry.connect_from(f"{original_shape}.{shape_output}")
        uv_pin_node.deformed_geometry.connect_from(f"{primary_shape}.{shape_output}")
        index = 0
    else:
        pin_indices = uv_pin_node.coordinate.get_indices()
        index = 0
        while index in pin_indices:
            index += 1

    normal_axis_enum = normal_axis if isinstance(normal_axis, Axis) else Axis.from_str(normal_axis)
    tangent_axis_enum = (
        tangent_axis if isinstance(tangent_axis, Axis) else Axis.from_str(tangent_axis)
    )

    uv_pin_node.normal_axis.set(normal_axis_enum)
    uv_pin_node.tangent_axis.set(tangent_axis_enum)
    uv_pin_node.normalized_isoparms.set(normalize)

    resolved_uv = _resolve_uv_for_pin(primary_shape, object_to_pin, uv, normalize)
    uv_pin_node.coordinate[index].set(resolved_uv)

    if drive_rotate or drive_translate:
        matrices: list[MatrixAttribute | MMatrix] = []
        if keep_offset:
            offset_matrix: MMatrix = (
                get_world_matrix(object_to_pin) * uv_pin_node.output_matrix[index].get().inverse()
            )
            matrices.append(offset_matrix)
        matrices.append(uv_pin_node.output_matrix[index])
        matrices.append(MatrixAttribute(f"{object_to_pin}.parentInverseMatrix[0]"))
        localize_matrix = multiply_matrices(f"{pin_name}_localize{index}", matrices)

        drive_transform_with_matrix(
            localize_matrix.matrix_sum,
            object_to_pin,
            translate=drive_translate,
            rotate=drive_rotate,
            scale=False,
            shear=False,
        )
    return uv_pin_node, index


def uv_pin_multi(
    name: str,
    surface: str,
    objects_to_pin: Iterable[str],
    uv_coords: Iterable[tuple[float, float]] | None = None,
    normalize: bool = False,
    normal_axis: Axis = Axis.Z,
    tangent_axis: Axis = Axis.X,
    keep_offset: bool = False,
) -> UvPinNode:
    """
    Pin multiple objects to a surface using a single shared uvPin node.

    Creates one UvPinNode and iterates over ``objects_to_pin`` pinning each.
    UV coordinates are matched positionally to objects; any object without a
    corresponding entry in ``uv_coords`` falls back to closest-point sampling.

    Args:
        name: Name for the created uvPin node.
        surface: The name of the surface (mesh or NURBS) to pin to.
        objects_to_pin: Ordered collection of object names to pin.
        uv_coords: Optional ordered collection of (u, v) pairs, matched
            positionally to ``objects_to_pin``. When fewer UV pairs than
            objects are provided the remaining objects are pinned to their
            closest point on the surface.
        normalize: Enable isoparm normalization (NURBS UV remapped to [0, 1]).
        normal_axis: Normal axis of the uvPin node, can be x y z -x -y -z.
        tangent_axis: Tangent axis of the uvPin node, can be x y z -x -y -z.
        keep_offset: When True, the pinned objects will be offset to be in
            the same world space placements as before being pinned.

    Returns:
        The shared UvPinNode with all objects registered as pin slots.
    """

    primary_shape, original_shape, shape_output = get_surface_shapes(surface)
    uv_pin_node = UvPinNode.create(name)
    uv_pin_node.original_geometry.connect_from(f"{original_shape}.{shape_output}")
    uv_pin_node.deformed_geometry.connect_from(f"{primary_shape}.{shape_output}")

    for object_to_pin, uv in zip_longest(
        objects_to_pin, uv_coords if uv_coords else (), fillvalue=None
    ):
        if object_to_pin is None:
            raise ValueError("More uv_coordinates than objects_to_pin. Unable to pin.")
        uv_pin(
            surface,
            object_to_pin,
            uv=uv,
            normalize=normalize,
            normal_axis=normal_axis,
            tangent_axis=tangent_axis,
            uv_pin_node=uv_pin_node,
            keep_offset=keep_offset,
        )

    return uv_pin_node


def closest_point_on_surface_reader(
    transform: str, surface: str
) -> ClosestPointOnSurfaceResultAttribute:
    transform_name = get_short_name(transform)
    shape = get_shape(surface)
    if shape is None:
        raise ValueError(f"{surface} has no valid shape")
    closest_point_node = ClosestPointOnSurfaceNode.create(f"{transform_name}_closestPoint")
    closest_point_node.input_surface.connect_from(f"{shape}.worldSpace[0]")

    world_driver_pos = MultiplyPointByMatrixNode.create(f"{transform_name}_world_pos")
    world_driver_pos.input_matrix.connect_from(f"{transform}.worldMatrix[0]")
    closest_point_node.in_position.connect_from(world_driver_pos.output)

    return closest_point_node.result


def surface_slide_constraint(
    surface: str,
    driver_transform: str,
    slider_transform: str,
    normal_axis: Axis = Axis.Z,
    secondary_axis: Axis = Axis.X,
    twist: bool = True,
    uv_pin_node: UvPinNode | None = None,
) -> None:
    """
    Constrain a slider transform to slide along a surface, driven by another transform.

    Args:
        surface: The name of the surface (mesh or NURBS) to slide along.
        driver_transform: The transform that drives the slide position. Its world
            position and rotation is projected onto the surface.
        slider_transform: The transform to be constrained. Its translation will
            follow the closest point on the surface to the driver.
        normal_axis: Local axis on ``slider_transform`` that should align with
            the surface normal. Defaults to (0, 0, 1) (Z-axis).
        secondary_axis: Local axis on ``slider_transform`` used as the up-vector
            for the normal constraint. Defaults to (0, 1, 0) (Y-axis).
    """

    slider_name = get_short_name(slider_transform)
    closest_point_reader = closest_point_on_surface_reader(driver_transform, surface)
    shape = get_shape(surface)
    if shape is None:
        raise ValueError(f"{surface} has no valid shape")

    resolved_uv_pin_node, pin_index = uv_pin(
        surface,
        slider_transform,
        normal_axis=normal_axis,
        tangent_axis=secondary_axis,
        uv_pin_node=uv_pin_node,
        drive_rotate=not twist,
    )
    resolved_uv_pin_node.coordinate[pin_index].u.connect_from(closest_point_reader.parameter_u)
    resolved_uv_pin_node.coordinate[pin_index].v.connect_from(closest_point_reader.parameter_v)

    if not twist:
        return

    normal_axis_tuple = normal_axis.to_tuple()
    secondary_axis_tuple = secondary_axis.to_tuple()

    matrix_normal_orient_constraint(
        resolved_uv_pin_node.output_matrix[pin_index],
        twist_transform=driver_transform,
        driven_transform=slider_transform,
        normal_axis=normal_axis_tuple,
        secondary_axis=secondary_axis_tuple,
    )
