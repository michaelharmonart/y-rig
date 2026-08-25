from __future__ import annotations

from dataclasses import dataclass

import maya.api.OpenMaya as om
from maya import cmds


@dataclass
class GuideLocator:
    """
    Stores information about a generated guide locator.
    """

    name: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float]
    matrix: list[float]


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
    ):

        self.input_curve = curve
        self.resample_amount = resample_amount
        self.output_names = output_names or []
        self.ignore_handles = ignore_handles
        self.align_normals = align_normals

        self.locator_list: list[GuideLocator] = []

        self.curve: str = ""
        self.group: str = ""
        self.count: int = 0

        self.build()

    # =========================================================
    # BUILD
    # =========================================================

    def build(self) -> None:

        self.duplicate_and_resample_curve()
        self.create_group()
        self.create_locators()

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

            loc: str = cmds.spaceLocator(name=loc_name)[0]  # type:ignore

            cmds.xform(
                loc,
                worldSpace=True,
                translation=pos,  # type:ignore
            )

            # -------------------------------------------------
            # Rotation
            # -------------------------------------------------

            rot = (0.0, 0.0, 0.0)

            if self.align_normals:
                rot = self._calculate_rotation_from_cvs(cvs, i)

                cmds.xform(
                    loc,
                    worldSpace=True,
                    rotation=rot,
                )

            cmds.parent(loc, self.group)

            # -------------------------------------------------
            # Store Data
            # -------------------------------------------------

            matrix = cmds.xform(
                loc,
                query=True,
                worldSpace=True,
                matrix=True,
            )

            locator_data = GuideLocator(
                name=loc,
                pos=tuple(pos),  # type:ignore
                rot=tuple(rot),
                matrix=matrix,  # type:ignore
            )

            self.locator_list.append(locator_data)

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
