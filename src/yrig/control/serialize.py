from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from maya import cmds
from maya.api.OpenMaya import (
    MDoubleArray,
    MFnNurbsCurve,
    MPointArray,
    MSelectionList,
    MSpace,
)

from yrig.control.utils import get_tagged_controls
from yrig.transform import create_transform, get_shapes
from yrig.util import confirm_overwrite

log = logging.getLogger(__name__)

SHAPE_LIBRARY_DIR = Path(Path(__file__).resolve().parent / "shape_library")
_control_shape_data_cache: dict[ControlShape, ControlShapeData] = {}


class ControlShape(Enum):
    """Enum for available control shapes with file names."""

    CIRCLE = "circle"
    SQUARE = "square"
    ROUND_SQUARE = "round_square"
    CUBE = "cube"
    SPHERE = "sphere"
    LOCATOR = "locator"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    LINE = "line"
    SEMI_CIRCLE = "semi_circle"

    @property
    def filename(self) -> str:
        """returns the filename of the json file representing the control shape."""
        return self.value


@dataclass(frozen=True)
class NurbsCurveData:
    degree: int
    form: int
    cv_positions: list[tuple[float, float, float]]
    cv_weights: list[float]
    knots: list[float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NurbsCurveData:
        return cls(
            degree=data["degree"],
            form=data["form"],
            cv_positions=[tuple(p) for p in data["cv_positions"]],
            cv_weights=data["cv_weights"],
            knots=data["knots"],
        )


@dataclass(frozen=True)
class NamedNurbsCurveData:
    name: str
    curve: NurbsCurveData


@dataclass(frozen=True)
class ControlShapeData:
    curves: list[NamedNurbsCurveData]

    def to_dict(self) -> dict:
        return {curve.name: curve.curve.to_dict() for curve in self.curves}

    @classmethod
    def from_dict(cls, data: dict) -> ControlShapeData:
        return cls(
            curves=[
                NamedNurbsCurveData(name, NurbsCurveData.from_dict(curve_data))
                for name, curve_data in data.items()
            ]
        )


def get_cv_data(curve_shape: str) -> tuple[list[tuple[float, float, float]], list[float]]:
    """
    Gets both the positions and weights of all CVs for a given curve shape.
    Args:
        curve_shape (str): Name of curve shape node.
    Returns:
        tuple: (positions, weights)
            positions (list[tuple[float, float, float]]): List of CV positions
            weights (list[float]): List of CV weights
    """
    sel: MSelectionList = MSelectionList()
    sel.add(curve_shape)
    curve_obj = sel.getDependNode(0)
    fn_curve: MFnNurbsCurve = MFnNurbsCurve(curve_obj)

    cv_positions: MPointArray = fn_curve.cvPositions(space=MSpace.kObject)
    positions: list[tuple[float, float, float]] = [
        (point.x, point.y, point.z) for point in cv_positions
    ]
    weights: list[float] = [point.w for point in cv_positions]

    return positions, weights


def get_knots(curve_shape: str) -> list[float]:
    """
    Gets the knot vector for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of knot values. (aka knot vector)
    """
    sel: MSelectionList = MSelectionList()
    sel.add(curve_shape)
    curve_obj = sel.getDependNode(0)
    fn_curve: MFnNurbsCurve = MFnNurbsCurve(curve_obj)

    knots_array: MDoubleArray = fn_curve.knots()
    knots: list[float] = [knot for knot in knots_array]
    return knots


def get_control_shape_data(curve: str) -> ControlShapeData:
    curves: list[NamedNurbsCurveData] = []
    for curve_shape in get_shapes(transform=curve):
        degree: int = cmds.getAttr(curve_shape + ".degree")
        form: int = cmds.getAttr(curve_shape + ".form")
        cv_positions: list[tuple[float, float, float]]
        cv_weights: list[float]
        cv_positions, cv_weights = get_cv_data(curve_shape=curve_shape)
        knots: list[float] = get_knots(curve_shape=curve_shape)
        curve_data = NurbsCurveData(degree, form, cv_positions, cv_weights, knots)
        curves.append(NamedNurbsCurveData(curve_shape, curve_data))
    return ControlShapeData(curves)


def control_shape_data_to_json(data: ControlShapeData) -> str:
    return json.dumps(data.to_dict(), indent=2)


def control_shape_data_from_json(json_str: str) -> ControlShapeData:
    data = json.loads(json_str)
    return ControlShapeData.from_dict(data)


def control_shape_data_from_library(curve_shape: ControlShape | str) -> ControlShapeData:
    """
    Args:
        curve_shape(ControlShape): Name of the control shape to retrieve.
    Returns:
        dict: Curve data.
    """
    if isinstance(curve_shape, str):
        curve_shape: ControlShape = ControlShape[curve_shape.strip().upper()]
    if curve_shape not in _control_shape_data_cache:
        # check if curve dict is a file and convert it to dictionary if it is
        file_path: Path = SHAPE_LIBRARY_DIR / f"{curve_shape.filename}.json"
        if not file_path.exists():
            raise RuntimeError(
                f"The shape file for {curve_shape.filename} couldn't be found in the shape library. "
                f"You must write out the file {file_path} before reading."
            )

        with open(file_path) as json_file:
            json_data = json_file.read()
            _control_shape_data_cache[curve_shape] = control_shape_data_from_json(json_data)
    return _control_shape_data_cache[curve_shape]


def create_shape_from_named_curve_data(
    named_curve: NamedNurbsCurveData, parent: str, use_name: bool = True
) -> str:
    curve = named_curve.curve
    positions: list[tuple[float, float, float]] = curve.cv_positions
    degree: int | list = curve.degree
    if isinstance(degree, (list, tuple)):
        degree = int(degree[0])
    else:
        degree = int(degree)

    form_val = curve.form
    if isinstance(form_val, (list, tuple)):
        form_val = int(form_val[0])
    else:
        form_val = int(form_val)

    periodic: bool = form_val == 2
    knots: list[float] = list(curve.knots)
    weights: list[float] = list(curve.cv_weights)
    position_weights: list[tuple[float, float, float, float]] = [
        (position[0], position[1], position[2], weights[index])
        for index, position in enumerate(positions)
    ]
    shape_name = named_curve.name
    try:
        child_curve_transform: str = cmds.curve(
            pointWeight=position_weights, knot=knots, periodic=periodic, degree=degree
        )
    except Exception:
        log.error(
            "Failed to create curve from data: degree=%r (type=%s), knots=%r, periodic=%r, position_weights_len=%d",
            degree,
            type(degree),
            knots if isinstance(knots, (list, tuple)) and len(knots) < 20 else "<long>",
            periodic,
            len(position_weights),
        )
        raise
    curve_shape_node: str = get_shapes(child_curve_transform)[0]
    if use_name:
        curve_shape_node = cmds.rename(curve_shape_node, shape_name)
    cmds.parent(curve_shape_node, parent, shape=True, relative=True)
    cmds.delete(child_curve_transform)
    return curve_shape_node


def create_curve_from_data(
    curve_data: ControlShapeData,
    name: str | None = None,
    parent: str | None = None,
) -> str:
    curve_transform: str = create_transform(name=name or "curve")
    for index, named_curve in enumerate(curve_data.curves):
        shape_node = create_shape_from_named_curve_data(
            named_curve, curve_transform, use_name=False
        )
        shape_name = f"{curve_transform}Shape" if index == 0 else f"{curve_transform}Shape{index}"
        cmds.rename(shape_node, shape_name)
    if parent is not None:
        cmds.parent(curve_transform, parent)
    if name is not None:
        curve_transform = cmds.rename(curve_transform, name)
    return curve_transform


def create_curve(
    name: str | None = None,
    control_shape: ControlShape | str = ControlShape.CIRCLE,
    parent: str | None = None,
) -> str:
    """
    Creates a curve from the specified item in the shape library.

    Args:
        curve_shape(ControlShape): Name of the control shape to generate.
    Returns:
        str: Name of the generated curve transform.
    """
    if isinstance(control_shape, str):
        control_shape: ControlShape = ControlShape[control_shape.strip().upper()]
    curve_data = control_shape_data_from_library(curve_shape=control_shape)
    curve_transform = create_curve_from_data(curve_data, name or control_shape.name)
    return curve_transform


def write_curve_to_library(
    curve: str | None = None, name: str | None = None, force: bool = False
) -> None:
    """
    Saves selected or defined curve to shape library.

    Args:
        control(str): Name of control to save. If None, uses the current selection.
        name(str): Name of the json file to save. If None, uses the chosen control.
    Returns:
        list: A list of CV weight values.
    """
    # make sure we either define a curve or have one selected
    # also make sure we're using the transform node
    if not curve:
        selection: list[str] = cmds.ls(selection=True)
        if len(selection) == 0:
            raise RuntimeError(
                "Unable to write control shape to file, no control transform was defined, and no control is selected."
            )
        curve: str = selection[0]

    # if a name is not defined, use the curves name instead
    if not name:
        name: str = curve

    json_path: Path = SHAPE_LIBRARY_DIR / f"{name}.json"
    if json_path.exists():
        if force:
            pass
        else:
            confirm: str = cmds.confirmDialog(
                title="File Overwrite",
                message=f"{json_path} already exists and will be overwritten, are you sure you want to write the file?",
                button=["Yes", "No"],
                defaultButton="Yes",
                cancelButton="No",
                dismissString="No",
            )
            if confirm == "Yes":
                pass
            else:
                log.info(
                    f"The control shape file for {curve} was not written as there was already a file present at:"
                    f"{json_path}"
                )
                return

    # get curve data
    curve_data = get_control_shape_data(curve=curve)
    json_dump = control_shape_data_to_json(curve_data)
    with open(file=json_path, mode="w") as json_file:
        json_file.write(json_dump)
    log.info(f"The control shape for {curve} was written to the shape library at {json_path}")


def export_control_shapes_file(filepath: Path, force: bool = False) -> bool:
    controls = get_tagged_controls()
    controls_shape_dict_data: dict[str, dict] = {
        control: get_control_shape_data(control).to_dict() for control in controls
    }
    if not confirm_overwrite(filepath, force):
        return False
    json_dump = json.dumps(controls_shape_dict_data, indent=2)
    with open(file=filepath, mode="w") as json_file:
        json_file.write(json_dump)
    log.info(f"Successfully exported control shapes file to {filepath}")
    return True


# Storing control data on the shape node is dumb, but for alwaysDrawOnTop it's the only way.
# Also mGear hooks things like visibility to the shape for some reason, so we need to maintain these.
@dataclass
class NurbsCurveShapeState:
    visibility: bool | str = True
    display_on_top: bool | str = False
    drawing_override: bool = False
    color_mode_rgb: bool = False
    color_index: int = 0
    color_rgb: tuple[float, float, float] = (0, 0, 0)
    color_alpha: float = 1

    def apply_to_nurbs_curve_shape(self, shape_node: str) -> None:
        default_state = NurbsCurveShapeState()
        if self.visibility != default_state.visibility:
            if isinstance(self.visibility, str):
                cmds.connectAttr(self.visibility, f"{shape_node}.visibility")
            else:
                cmds.setAttr(f"{shape_node}.visibility", self.visibility)  # type: ignore
        if self.display_on_top != default_state.display_on_top:
            if isinstance(self.display_on_top, str):
                cmds.connectAttr(self.display_on_top, f"{shape_node}.alwaysDrawOnTop")
            else:
                cmds.setAttr(f"{shape_node}.alwaysDrawOnTop", self.display_on_top)  # type: ignore
        if self.drawing_override != default_state.drawing_override:
            cmds.setAttr(f"{shape_node}.overrideEnabled", self.drawing_override)  # type: ignore
        if self.color_mode_rgb != default_state.drawing_override:
            cmds.setAttr(f"{shape_node}.drawOverride.overrideRGBColors", self.color_mode_rgb)  # type: ignore
        if self.color_index != default_state.color_index:
            cmds.setAttr(
                f"{shape_node}.drawOverride.overrideColor",
                self.color_index,  # type : ignore
            )
        if self.color_rgb != default_state.color_rgb:
            cmds.setAttr(
                f"{shape_node}.drawOverride.overrideColorRGB",
                *self.color_rgb,  # type : ignore
                type="float3",
            )
        if self.color_alpha != default_state.color_alpha:
            cmds.setAttr(f"{shape_node}.drawOverride.overrideColorA", self.color_alpha)  # type: ignore

    @classmethod
    def from_nurbs_curve_shape(cls, shape_node: str) -> NurbsCurveShapeState:
        visibility_attr = f"{shape_node}.visibility"
        visibility_connections = cmds.listConnections(
            visibility_attr, source=True, destination=False, plugs=True
        )
        if visibility_connections:
            visibility = visibility_connections[0]
        else:
            visibility = bool(cmds.getAttr(visibility_attr))

        display_on_top_attr = f"{shape_node}.alwaysDrawOnTop"
        display_on_top_connections = cmds.listConnections(
            display_on_top_attr, source=True, destination=False, plugs=True
        )
        if display_on_top_connections:
            display_on_top = display_on_top_connections[0]
        else:
            display_on_top = bool(cmds.getAttr(display_on_top_attr))

        drawing_override = bool(cmds.getAttr(f"{shape_node}.overrideEnabled"))
        color_mode_rgb = bool(cmds.getAttr(f"{shape_node}.drawOverride.overrideRGBColors"))
        color_index = int(cmds.getAttr(f"{shape_node}.drawOverride.overrideColor"))
        color_rgb: tuple[float, float, float] = cmds.getAttr(
            f"{shape_node}.drawOverride.overrideColorRGB"
        )[0]
        color_alpha: float = cmds.getAttr(f"{shape_node}.drawOverride.overrideColorA")
        return NurbsCurveShapeState(
            visibility,
            display_on_top,
            drawing_override,
            color_mode_rgb,
            color_index,
            color_rgb,
            color_alpha,
        )


def apply_control_shape_data(control: str, data: ControlShapeData) -> None:
    old_control_shapes = get_shapes(control)
    # Just pick one shape that'll be the source for shape state
    primary_shape = old_control_shapes[0]
    shape_state = NurbsCurveShapeState.from_nurbs_curve_shape(primary_shape)
    # Rename the old shapes so that there aren't name conflicts. We can't delete yet as Maya might delete connected nodes.
    old_control_shapes = [
        cmds.rename(control_shape, f"{control_shape}_old") for control_shape in old_control_shapes
    ]
    for shape_data in data.curves:
        shape_node = create_shape_from_named_curve_data(shape_data, control)
        shape_state.apply_to_nurbs_curve_shape(shape_node)
    cmds.delete(old_control_shapes)  # type: ignore


def apply_control_shapes_file(filepath: Path) -> None:
    if not filepath.exists():
        raise RuntimeError(f"There was no control shapes file found at {filepath}")

    existing_controls: set[str] = set(get_tagged_controls())
    control_dict: dict[str, dict]
    with open(filepath) as json_file:
        json_data = json_file.read()
        control_dict = json.loads(json_data)
    for control, control_shape_data_dict in control_dict.items():
        if control not in existing_controls:
            continue
        control_shape_data = ControlShapeData.from_dict(control_shape_data_dict)
        apply_control_shape_data(control, control_shape_data)
    log.info(f"Control shapes loaded and applied from {filepath}")
