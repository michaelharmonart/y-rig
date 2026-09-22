from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from maya import cmds
from maya.api.OpenMaya import MMatrix

from yrig.build.mgear_api.control import add_ctl
from yrig.color.convert import (
    byte_color_to_rgb,
    hex_to_byte_color,
    srgb_to_linear_color,
)
from yrig.control.serialize import ControlShape, create_curve
from yrig.maya_api.enum import RotateOrder
from yrig.name import MIDDLE_SIDE_NAME, get_side
from yrig.shape import bake_shape
from yrig.transform import create_transform, get_shapes, partial_path_name
from yrig.transform.matrix import get_world_matrix
from yrig.transform.structs import Direction

CONTROL_SUFFIX = "_ctl"
OFFSET_SUFFIX = "_npo"

_control_collection: ContextVar[list[Control] | None] = ContextVar(
    "control_collection", default=None
)


@contextmanager
def collect_controls() -> Iterator[list[Control]]:
    """
    Collect controls created inside this block.

    Any controls created within the `with` statement are added to a list,
    which is returned when the block finishes. Nested blocks are supported,
    and inner results are included in the outer list.

    Returns:
        list[Control]: Controls created in the block.
    """
    # Create a bucket to collect the controls created in the with block
    # then put it into the ContextVar so that _register_control will add to this bucket
    bucket: list[Control] = []
    parent_bucket = _control_collection.get()
    token = _control_collection.set(bucket)
    try:
        yield bucket
    finally:
        # Restore the previous state
        _control_collection.reset(token)
        # If there was a parent, bubble up the results
        if parent_bucket is not None:
            parent_bucket.extend(bucket)


def _register_control(ctrl: Control) -> None:
    bucket = _control_collection.get()
    if bucket is not None:
        bucket.append(ctrl)


@dataclass
class Control:
    transform: str
    offset: str
    name: str

    def __str__(self) -> str:
        return self.transform


def _create_control_curve(
    name: str,
    control_shape: ControlShape | str = ControlShape.CIRCLE,
    direction: Direction = "y",
    size: float = 1,
    dimensions: tuple[float, float, float] = (1, 1, 1),
    position_offset: tuple[float, float, float] = (0, 0, 0),
) -> str:
    curve_transform = create_curve(name, control_shape)
    bake: bool = False
    match direction:
        case "y":
            pass
        case "-y":
            cmds.rotate(180, 0, 0, curve_transform)
            bake = True
        case "x":
            cmds.rotate(0, 0, -90, curve_transform)
            bake = True
        case "-x":
            cmds.rotate(0, 0, 90, curve_transform)
            bake = True
        case "z":
            cmds.rotate(90, 0, 0, curve_transform)
            bake = True
        case "-z":
            cmds.rotate(-90, 0, 0, curve_transform)
            bake = True
        case _:
            raise RuntimeError(
                f"{direction} is not a valid direction. It should be x,y,z or -x,-y,-z."
            )
    if position_offset != (0, 0, 0):
        cmds.move(position_offset[0], position_offset[1], position_offset[2], curve_transform)
        bake = True

    if (size != 1) or (dimensions != (1, 1, 1)):
        scaled_dimensions = (size * dimension for dimension in dimensions)
        cmds.scale(*scaled_dimensions, curve_transform, relative=False)  # type: ignore
        bake = True

    if bake:
        bake_shape(transform=curve_transform)
    return curve_transform


def create_control(
    name: str,
    parent: str | Control | None,
    transform: str | MMatrix | None = None,
    control_shape: ControlShape | str = ControlShape.CIRCLE,
    direction: Direction = "y",
    size: float = 1,
    dimensions: tuple[float, float, float] = (1, 1, 1),
    rotation_order: RotateOrder = RotateOrder.XYZ,
    limit_min_scale: bool = True,
    position_offset: tuple[float, float, float] = (0, 0, 0),
) -> Control:
    transform_matrix: MMatrix | None
    if transform is not None:
        if isinstance(transform, str):
            transform_matrix = get_world_matrix(transform)
        elif isinstance(transform, MMatrix):
            transform_matrix = transform
        else:
            raise RuntimeError(f"{transform} is not a valid transform name or MMatrix")
    else:
        transform_matrix = None
    parent_transform = parent.transform if isinstance(parent, Control) else parent

    offset_transform = create_transform(
        name=f"{name}{OFFSET_SUFFIX}", parent=parent_transform, transform=transform_matrix
    )

    control_parent = offset_transform
    control_name = f"{name}{CONTROL_SUFFIX}"
    # We call a function to create an mGear compatible control here, since mGear is rather specific about what it needs.
    # Feel free to replace this if you ditch mGear.
    control_transform_path = str(
        add_ctl(
            control_name,
            control_parent,
            None,
            side=get_side(name) or MIDDLE_SIDE_NAME,
            control_icon_creator=lambda: _create_control_curve(
                control_name, control_shape, direction, size, dimensions, position_offset
            ),
            rotation_order=str(rotation_order),
        )
    )
    control_transform = partial_path_name(control_transform_path)
    cmds.setAttr(f"{control_transform}.visibility", keyable=False, channelBox=False)
    if limit_min_scale:  # Comfort feature: make it so it's not possible to have negative scale
        min_scale: float = 0.01
        cmds.transformLimits(
            control_transform,
            enableScaleX=(True, False),
            scaleX=(min_scale, 1),
            enableScaleY=(True, False),
            scaleY=(min_scale, 1),
            enableScaleZ=(True, False),
            scaleZ=(min_scale, 1),
        )

    control = Control(transform=control_transform, offset=offset_transform, name=name)
    _register_control(control)
    return control


def set_override_color(
    control: str | Control, color: tuple[float, float, float] | str, override_shapes: bool = False
) -> None:
    if isinstance(color, str):
        resolved_color = srgb_to_linear_color(byte_color_to_rgb(hex_to_byte_color(color)))
    else:
        resolved_color = color
    if override_shapes:
        override_nodes = [shape for shape in get_shapes(str(control))]
    else:
        override_nodes = (str(control),)

    for node in override_nodes:
        cmds.setAttr(f"{node}.overrideEnabled", True)  # type: ignore
        cmds.setAttr(f"{node}.overrideRGBColors", True)  # type: ignore
        cmds.setAttr(f"{node}.overrideColorRGB", *resolved_color)  # type: ignore
