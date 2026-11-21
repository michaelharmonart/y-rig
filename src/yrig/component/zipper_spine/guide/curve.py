from __future__ import annotations

from typing import Literal

import maya.api.OpenMaya as om
from maya import cmds
from maya.api.OpenMaya import (
    MFnNurbsCurve,
    MMatrix,
    MPoint,
    MSelectionList,
    MSpace,
    MVector,
)

from .core import GuideInfo, create_guide_from_position

Axis = Literal["x", "y", "z"]
ParameterMode = Literal["normalized", "parameter"]
CurveSpace = Literal["world", "object"]


class GuideCurve:
    """
    Utility class for reading and rebuilding guide curves.

    Example:
        guide_object = GuideCurve(
            curve="spine_guide_crv",
            resample_amount=5,
            output_names=["hip", "spineA", "spineB"],
            ignore_handles=True,
            align_normals=True,
        )

        print(guide_object.locator_list[0].name)
        print(guide_object.locator_list[0].pos)
        print(guide_object.locator_list[0].rot)
        print(guide_object.locator_list[0].matrix)

        print(guide_object.curve)
        print(guide_object.group)
        print(guide_object.count)
    """

    def __init__(
        self,
        curve: str,
        resample_amount: int = -1,
        output_names: list[str] | None = None,
        ignore_handles: bool = True,
        align_normals: bool = False,
        mirror: bool = False,
        primary_axis: str | None = None,
    ):

        self.input_curve = curve
        self.resample_amount = resample_amount
        self.output_names = output_names or []
        self.ignore_handles = ignore_handles
        self.align_normals = align_normals
        self.mirror = mirror
        self.primary_axis = primary_axis

        self.locator_list: list[GuideInfo] = []

        self.curve: str = ""
        self.group: str = ""
        self.count: int = 0

        self.build()

    # =========================================================
    # BUILD
    # =========================================================

    def refresh_locator_data(self) -> None:
        """
        Update stored locator positions, rotations,
        and world matrices after mirroring.
        """

        for locator in self.locator_list:
            locator.pos = tuple(
                cmds.xform(
                    locator.name,
                    query=True,
                    worldSpace=True,
                    translation=True,
                )  # type:ignore
            )

            locator.rot = tuple(
                cmds.xform(
                    locator.name,
                    query=True,
                    worldSpace=True,
                    rotation=True,
                )  # type:ignore
            )

            matrix = cmds.xform(
                locator.name,
                query=True,
                worldSpace=True,
                matrix=True,
            )

            if not isinstance(matrix, (list, tuple)):
                raise TypeError(f"Expected matrix, got {type(matrix).__name__}")

            locator.matrix = [float(value) for value in matrix]  # type:ignore

    def mirror_curve_for_build(self) -> None:
        """
        Reflect the duplicated curve across world X = 0
        before creating guides.
        """

        mirror_grp = cmds.group(
            empty=True,
            name=f"{self.input_curve}_tempMirror_grp",
        )

        cmds.parent(
            self.curve,
            mirror_grp,
        )

        cmds.setAttr(
            f"{mirror_grp}.scaleX",
            -1,  # type:ignore
        )

        cmds.parent(
            self.curve,
            world=True,
        )

        cmds.delete(mirror_grp)

    def mirror_guide_group(self) -> None:
        """
        Mirror the completed guide group back across X.

        The negative scale is retained on the group so
        its children inherit the reflected transform.
        """

        cmds.setAttr(
            f"{self.group}.scaleX",
            -1,  # type:ignore
        )

    def enforce_curve_direction(self) -> bool:
        """
        Ensure the curve starts closest to the world X center
        and progresses outward.

        Works for both positive and negative X.

        Returns:
            bool: True if the curve was reversed.
        """

        cvs = cmds.ls(
            f"{self.curve}.cv[*]",
            flatten=True,
        )

        if len(cvs) < 2:
            return False

        start_pos = cmds.pointPosition(
            cvs[0],
            world=True,
        )

        end_pos = cmds.pointPosition(
            cvs[-1],
            world=True,
        )

        start_distance = abs(start_pos[0])
        end_distance = abs(end_pos[0])

        if start_distance > end_distance:
            cmds.reverseCurve(
                self.curve,
                constructionHistory=False,
                replaceOriginal=True,
            )

            return True

        return False

    def ensure_primary_axis(self) -> None:
        """Optionally orient the duplicated curve along a signed world axis."""
        if self.primary_axis is None:
            return

        axis = self.primary_axis.upper().strip()
        axis_indices = {
            "+X": (0, 1),
            "-X": (0, -1),
            "+Y": (1, 1),
            "-Y": (1, -1),
            "+Z": (2, 1),
            "-Z": (2, -1),
        }
        if axis not in axis_indices:
            raise ValueError(
                f"Invalid primary_axis: {self.primary_axis}. Expected +X, -X, +Y, -Y, +Z, or -Z."
            )

        cvs = cmds.ls(f"{self.curve}.cv[*]", flatten=True) or []
        if len(cvs) < 2:
            return

        axis_index, desired_sign = axis_indices[axis]
        start = cmds.pointPosition(cvs[0], world=True)
        end = cmds.pointPosition(cvs[-1], world=True)
        delta = end[axis_index] - start[axis_index]
        if abs(delta) < 1e-6:
            return

        if (1 if delta > 0 else -1) != desired_sign:
            cmds.reverseCurve(
                self.curve,
                constructionHistory=False,
                replaceOriginal=True,
            )

    def build(self) -> None:

        self.duplicate_and_resample_curve()

        if self.mirror:
            self.mirror_curve_for_build()

        if self.primary_axis is None:
            self.enforce_curve_direction()
        else:
            self.ensure_primary_axis()

        self.create_group()
        self.create_locators()

        if self.mirror:
            self.mirror_guide_group()
            self.refresh_locator_data()

        self.count = len(self.locator_list)

    # =========================================================
    # CURVE SETUP
    # =========================================================

    def duplicate_and_resample_curve(self) -> None:

        self.curve = cmds.duplicate(
            self.input_curve,
            name=f"{self.input_curve}_guideCurve",
        )[0]

        if self.resample_amount != -1:
            cmds.rebuildCurve(
                self.curve,
                constructionHistory=False,
                replaceOriginal=True,
                rebuildType=0,
                endKnots=1,
                keepRange=0,
                keepControlPoints=False,
                keepEndPoints=True,
                keepTangents=False,
                spans=self.resample_amount - 1,
                degree=3,
                tolerance=0.01,
            )

    def create_group(self) -> None:

        self.group = cmds.group(
            empty=True,
            name=f"{self.input_curve}_guide_grp",
        )

        cmds.parent(self.curve, self.group)

    # =========================================================
    # LOCATORS
    # =========================================================

    def create_locators(self) -> None:

        cvs = cmds.ls(f"{self.curve}.cv[*]", flatten=True)

        if self.ignore_handles and len(cvs) > 4:
            cvs = cvs[1:-1]

        padding = len(str(len(cvs)))

        for i, cv in enumerate(cvs):
            pos: list[float] = cmds.pointPosition(cv, world=True)

            # -------------------------------------------------
            # Name
            # -------------------------------------------------

            if i < len(self.output_names):
                loc_name = self.output_names[i]
            else:
                loc_name = f"{self.input_curve}_cv_{str(i + 1).zfill(padding)}"

            guide = create_guide_from_position(
                pos=pos,
                guide_name=loc_name,
                parent=self.group,
            )

            # Align the actual Workshop guide before caching its data.
            if self.align_normals:
                rot = self._calculate_rotation_from_cvs(cvs, i)
                cmds.xform(
                    guide.name,
                    worldSpace=True,
                    rotation=rot,
                )

            self.locator_list.append(guide)

    # =========================================================
    # NORMAL / ROTATION
    # =========================================================

    def _calculate_rotation_from_cvs(
        self,
        cvs: list[str],
        index: int,
    ) -> tuple[float, float, float]:

        current_pos = om.MVector(cmds.pointPosition(cvs[index], world=True))

        # -----------------------------------------------------
        # Edge Cases
        # -----------------------------------------------------

        if index == 0:
            next_pos = om.MVector(cmds.pointPosition(cvs[index + 1], world=True))

            tangent = (next_pos - current_pos).normalize()

        elif index == len(cvs) - 1:
            prev_pos = om.MVector(cmds.pointPosition(cvs[index - 1], world=True))

            tangent = (current_pos - prev_pos).normalize()

        else:
            prev_pos = om.MVector(cmds.pointPosition(cvs[index - 1], world=True))

            next_pos = om.MVector(cmds.pointPosition(cvs[index + 1], world=True))

            tangent = (next_pos - prev_pos).normalize()

        # -----------------------------------------------------
        # Build Rotation Matrix
        # -----------------------------------------------------

        up_vector = om.MVector(0, 1, 0)

        # Prevent parallel vector issue
        if abs(tangent * up_vector) > 0.999:
            up_vector = om.MVector(1, 0, 0)

        side = (tangent ^ up_vector).normalize()
        up = (side ^ tangent).normalize()

        matrix = om.MMatrix(
            [
                side.x,
                side.y,
                side.z,
                0.0,
                up.x,
                up.y,
                up.z,
                0.0,
                tangent.x,
                tangent.y,
                tangent.z,
                0.0,
                current_pos.x,
                current_pos.y,
                current_pos.z,
                1.0,
            ]
        )

        transform_matrix = om.MTransformationMatrix(matrix)

        euler = transform_matrix.rotation()

        rot = (
            om.MAngle(euler.x).asDegrees(),
            om.MAngle(euler.y).asDegrees(),
            om.MAngle(euler.z).asDegrees(),
        )

        return rot


def create_line_curve(
    name: str,
    start_position: tuple[float, float, float],
    end_position: tuple[float, float, float],
) -> str:
    """Create a straight curve with duplicated endpoint CVs."""

    return cmds.curve(
        name=name,
        degree=3,
        point=[
            start_position,
            start_position,
            end_position,
            end_position,
        ],
    )


def style_curve(
    curve: str, line_width: float = 5.0, draw_on_top: bool = False, template: bool = True
) -> None:
    """Style a curve for guide display."""

    shapes = cmds.listRelatives(curve, shapes=True, fullPath=True) or []

    for shape in shapes:
        cmds.setAttr(f"{shape}.lineWidth", line_width)  # type:ignore
        cmds.setAttr(f"{shape}.alwaysDrawOnTop", draw_on_top)  # type:ignore
        cmds.setAttr(f"{shape}.template", template)  # type:ignore


def curve_cvs(curve: str, clusters: bool = False) -> tuple[list]:
    """Create one cluster per curve CV.

    Returns:
        A 2 lists containing:
        [
            (cluster_handle, cv_world_position),
            ...
        ]
    """

    cvs = cmds.ls(f"{curve}.cv[*]", flatten=True) or []
    cluster_data = []
    cv_pos = []
    cluster_nodes = []

    for index, cv in enumerate(cvs):
        position = cmds.xform(
            cv,
            query=True,
            worldSpace=True,
            translation=True,
        )

        if clusters:
            cluster_node, cluster_handle = cmds.cluster(
                cv,
                name=f"{curve}_cv{index}_cluster",
            )  # type:ignore
            cluster_data.append(cluster_handle)
            cluster_nodes.append(cluster_node)

        cv_pos.append(position)

    return cluster_data, cv_pos  # type:ignore


def are_curves_mirrored(
    curve_a: str,
    curve_b: str,
    axis: str = "x",
    tolerance: float = 1e-4,
    allow_reversed: bool = True,
) -> bool:
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Invalid mirror axis: {axis}")

    def curve_fn(curve: str) -> om.MFnNurbsCurve:
        selection = om.MSelectionList()
        selection.add(get_curve_shape(curve))
        return om.MFnNurbsCurve(selection.getDagPath(0))

    a, b = curve_fn(curve_a), curve_fn(curve_b)
    if (a.numCVs, a.degree, a.form) != (b.numCVs, b.degree, b.form):
        return False
    points_a = a.cvPositions(om.MSpace.kWorld)
    points_b = b.cvPositions(om.MSpace.kWorld)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]

    def matches(points: list) -> bool:
        for p_a, p_b in zip(points_a, points, strict=False):
            mirrored = om.MPoint(p_a)
            mirrored[axis_index] *= -1
            if mirrored.distanceTo(p_b) > tolerance:
                return False
        return True

    return matches(points_b) or (allow_reversed and matches(reversed(points_b)))


def get_curve_shape(curve: str) -> str:
    if not cmds.objExists(curve):
        raise ValueError(f"Curve does not exist: {curve}")
    if cmds.nodeType(curve) == "nurbsCurve":
        return curve
    shapes = cmds.listRelatives(curve, shapes=True, noIntermediate=True, fullPath=True) or []
    for shape in shapes:
        if cmds.nodeType(shape) == "nurbsCurve":
            return shape
    raise ValueError(f"Object does not contain a NURBS curve: {curve}")


def resample_curve(curve: str, spans: int = 3, degree: int = 3) -> str:
    get_curve_shape(curve)
    if spans < 1:
        raise ValueError("Spans must be at least 1.")
    if degree not in (1, 2, 3):
        raise ValueError("Degree must be 1, 2, or 3.")
    cmds.rebuildCurve(
        curve,
        ch=False,  # type:ignore
        rpo=True,  # type:ignore
        rt=0,  # type:ignore
        end=1,  # type:ignore
        kr=0,  # type:ignore
        kcp=False,  # type:ignore
        kep=True,  # type:ignore
        kt=False,  # type:ignore
        s=max(spans, degree),  # type:ignore
        d=degree,  # type:ignore
    )
    return curve


def reverse_curve(curve: str) -> str:
    get_curve_shape(curve)
    cmds.reverseCurve(curve, ch=False, rpo=True)  # type:ignore
    return curve


def orient_curve(
    curve: str,
    direction: str = "outward",
    axis: str = "X",
    reverse: bool = False,
    center: float = 0.0,
) -> str:
    """Orient an OPEN curve by its world-space endpoints.

    outward/inward use distance from the origin's symmetry plane on axis.
    positive/negative use signed axis travel; preserve leaves the order alone.
    reverse=True flips the final result, including preserve.
    """
    axis = axis.upper()
    direction = direction.lower()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"Invalid axis: {axis}")
    if direction not in {"outward", "inward", "positive", "negative", "preserve"}:
        raise ValueError(f"Invalid direction: {direction}")
    shape = get_curve_shape(curve)
    if cmds.getAttr(f"{shape}.form") != 0:
        if direction != "preserve" or reverse:
            raise ValueError(
                "Direction orientation requires an open curve; use preserve for closed curves."
            )
        return curve
    cvs = cmds.ls(f"{curve}.cv[*]", flatten=True) or []
    if len(cvs) < 2:
        return curve
    i = {"X": 0, "Y": 1, "Z": 2}[axis]
    start = cmds.pointPosition(cvs[0], world=True)[i] - center
    end = cmds.pointPosition(cvs[-1], world=True)[i] - center
    should_reverse = False
    if direction == "outward":
        should_reverse = abs(start) > abs(end) + 1e-6
    elif direction == "inward":
        should_reverse = abs(start) + 1e-6 < abs(end)
    elif direction == "positive":
        should_reverse = start > end + 1e-6
    elif direction == "negative":
        should_reverse = start + 1e-6 < end
    if should_reverse != reverse:
        reverse_curve(curve)
    return curve


def create_curve_through_positions(
    name: str,
    positions: list[tuple[float, float, float]],
    degree: int = 3,
) -> str:
    """Create an open NURBS curve interpolating the given world-space positions.

    Maya's editPoint flag places the curve THROUGH the points, rather than
    treating them as control vertices. Degree is clamped to the point count.
    """
    if len(positions) < 2:
        raise ValueError("At least two positions are required")
    if degree < 1:
        raise ValueError("Degree must be at least 1")
    actual_degree = min(degree, len(positions) - 1)
    return cmds.curve(
        name=name,
        degree=actual_degree,
        editPoint=[tuple(point) for point in positions],
    )


def get_nurbs_curve_radius(curve: str) -> float:
    """
    Returns the radius of a NURBS circle.

    Args:
        obj (str): Transform of the NURBS circle

    Returns:
        float: radius
    """

    bbox: list[int | float] = cmds.exactWorldBoundingBox(curve)

    x: int | float = bbox[3] - bbox[0]
    y: int | float = bbox[4] - bbox[1]
    z: int | float = bbox[5] - bbox[2]

    return max(x, y, z) * 0.5


def smooth_curve_weights(
    curve: str,
    iterations: int = 5,
    strength: float = 0.5,
    preserve_ends: bool = True,
) -> None:
    """Smooth skin weights along a NURBS curve."""

    strength = max(0.0, min(1.0, strength))

    history = cmds.listHistory(curve) or []
    skins = cmds.ls(history, type="skinCluster")  # type:ignore

    if not skins:
        raise ValueError(f"No skinCluster found on {curve}")

    skin = skins[0]

    influences = cmds.skinCluster(
        skin,
        query=True,
        influence=True,
    )

    cvs = cmds.ls(
        f"{curve}.cv[*]",
        flatten=True,
    )

    if not cvs:
        raise ValueError(f"No CVs found on {curve}")

    for _ in range(iterations):
        weights = [
            [
                cmds.skinPercent(
                    skin,
                    cv,
                    query=True,
                    transform=joint,
                )
                for joint in influences  # type:ignore
            ]
            for cv in cvs
        ]

        for i, cv in enumerate(cvs):
            if preserve_ends and i in (0, len(cvs) - 1):
                continue

            neighbors = [weights[j] for j in (i - 1, i + 1) if 0 <= j < len(cvs)]

            if not neighbors:
                continue

            new_weights = []

            for j, weight in enumerate(weights[i]):
                average = sum(neighbor[j] for neighbor in neighbors) / len(neighbors)  # type:ignore

                smoothed = weight * (1.0 - strength) + average * strength  # type:ignore

                new_weights.append(smoothed)

            cmds.skinPercent(
                skin,
                cv,
                transformValue=list(zip(influences, new_weights, strict=True)),  # type:ignore
                normalize=True,
            )


def get_curve_point_matrix(
    curve: str,
    parameter: float,
    tangent_orient: bool = True,
    up_vector: tuple[float, float, float] = (0, 1, 0),
    aim_axis: Axis = "x",
    mirror: bool = False,
    mirror_axis: Axis = "x",
    parameter_mode: ParameterMode = "normalized",
    space: CurveSpace = "world",
) -> MMatrix:
    """
    Get a matrix at a point along a NURBS curve.

    Args:
        curve: Curve transform or shape.
        parameter: Position along the curve.
        tangent_orient: Align orientation to the curve tangent.
        up_vector: Reference up direction in the requested space.
        aim_axis: Local axis aligned with the tangent.
        mirror: Construct a mirrored orientation.
        mirror_axis: Axis used for reflection.
        parameter_mode:
            'normalized': Arc-length fraction (0-1).
            'parameter': Native NURBS curve parameter.
        space:
            'world': Return a world-space matrix.
            'object': Return a curve-local matrix.

    Returns:
        MMatrix containing translation and orientation.
    """

    # --------------------------------------------------
    # Resolve curve
    # --------------------------------------------------

    if cmds.nodeType(curve) == "nurbsCurve":
        shape = curve
    else:
        shapes = (
            cmds.listRelatives(
                curve,
                shapes=True,
                noIntermediate=True,
                type="nurbsCurve",
                fullPath=True,
            )
            or []
        )

        if not shapes:
            raise ValueError(f"No NURBS curve found: {curve}")

        shape = shapes[0]

    selection = MSelectionList()
    selection.add(shape)

    dag_path = selection.getDagPath(0)
    fn_curve = MFnNurbsCurve(dag_path)

    # --------------------------------------------------
    # Resolve sampling space and parameter
    # --------------------------------------------------

    sample_space = MSpace.kWorld if space == "world" else MSpace.kObject

    if parameter_mode == "normalized":
        parameter = max(0.0, min(1.0, parameter))

        curve_param = fn_curve.findParamFromLength(fn_curve.length() * parameter)

    elif parameter_mode == "parameter":
        curve_param = parameter

    else:
        raise ValueError(f"Unsupported parameter mode: {parameter_mode}")

    position = fn_curve.getPointAtParam(
        curve_param,
        sample_space,
    )

    # --------------------------------------------------
    # Reflection helper
    # --------------------------------------------------

    if mirror_axis not in ("x", "y", "z"):
        raise ValueError(f"Invalid mirror axis: {mirror_axis}")

    axis_index = {"x": 0, "y": 1, "z": 2}[mirror_axis]

    def reflect(vector: MVector) -> MVector:
        values = [vector.x, vector.y, vector.z]
        values[axis_index] *= -1.0
        return MVector(*values)

    if mirror:
        mirrored_position = reflect(MVector(position.x, position.y, position.z))
        position = MPoint(
            mirrored_position.x,
            mirrored_position.y,
            mirrored_position.z,
        )

    # --------------------------------------------------
    # Construct orientation
    # --------------------------------------------------

    if tangent_orient:
        tangent = fn_curve.tangent(
            curve_param,
            sample_space,
        ).normal()

        if mirror:
            tangent = reflect(tangent)

        aim = tangent.normal()
        up = MVector(*up_vector).normal()

        if aim.length() < 1e-8:
            raise ValueError("Curve tangent has zero length")

        if up.length() < 1e-8:
            raise ValueError("Up vector cannot be zero")

        # Avoid parallel vectors.
        if abs(aim * up) > 0.999:
            candidates = (
                MVector(1, 0, 0),
                MVector(0, 1, 0),
                MVector(0, 0, 1),
            )

            up = min(
                candidates,
                key=lambda axis: abs(aim * axis),
            )

        side = (aim ^ up).normal()
        up = (side ^ aim).normal()

        if aim_axis == "x":
            x_axis = aim
            y_axis = up
            z_axis = side

        elif aim_axis == "y":
            x_axis = -side
            y_axis = aim
            z_axis = up

        elif aim_axis == "z":
            x_axis = up
            y_axis = -side
            z_axis = aim

        else:
            raise ValueError(f"Invalid aim axis: {aim_axis}")

    else:
        x_axis = MVector(1, 0, 0)
        y_axis = MVector(0, 1, 0)
        z_axis = MVector(0, 0, 1)

    # --------------------------------------------------
    # Build matrix
    # --------------------------------------------------

    matrix = MMatrix(
        [
            x_axis.x,
            x_axis.y,
            x_axis.z,
            0.0,
            y_axis.x,
            y_axis.y,
            y_axis.z,
            0.0,
            z_axis.x,
            z_axis.y,
            z_axis.z,
            0.0,
            position.x,
            position.y,
            position.z,
            1.0,
        ]
    )

    # --------------------------------------------------
    # Apply reflection
    # --------------------------------------------------

    if mirror:
        scale = [1.0, 1.0, 1.0]
        scale[axis_index] = -1.0

        reflection = MMatrix(
            [
                scale[0],
                0.0,
                0.0,
                0.0,
                0.0,
                scale[1],
                0.0,
                0.0,
                0.0,
                0.0,
                scale[2],
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

        matrix = matrix * reflection

    return matrix
