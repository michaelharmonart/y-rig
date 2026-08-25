from typing import Literal

from maya import cmds

from yrig.control import Control, create_control
from yrig.maya_api.attribute import BooleanAttribute, ScalarAttribute
from yrig.maya_api.node import MultiplyNode
from yrig.surface import surface_slide_constraint


class MouthCorner:
    def __init__(
        self,
        side: Literal["L", "R"],
        guide: str,
        mouth_surface: str,
        control_parent: Control | str,
        control_size: float = 1,
        sub_control_vis_attr: BooleanAttribute | None = None,
    ):
        self.main_control = create_control(
            f"mouth_corner_{side}",
            transform=guide,
            parent=control_parent,
            size=control_size,
            direction="z",
        )
        self.sub_control = create_control(
            f"mouth_corner_{side}_sub",
            transform=guide,
            parent=self.main_control,
            size=control_size * 0.5,
            direction="z",
        )

        surface_slide_constraint(
            mouth_surface,
            driver_transform=self.main_control.transform,
            slider_transform=self.sub_control.offset,
        )

        self.upper_control = create_control(
            f"mouth_corner_{side}_up",
            transform=self.sub_control.offset,
            parent=self.sub_control.offset,
            size=control_size * 0.5,
            direction="z",
        )
        self.lower_control = create_control(
            f"mouth_corner_{side}_lo",
            transform=self.sub_control.offset,
            parent=self.sub_control.offset,
            size=control_size * 0.5,
            direction="z",
        )
        for control in (self.upper_control, self.lower_control):
            cmds.setAttr(f"{control.transform}.translateZ", lock=True)

        self.roundness_attr = ScalarAttribute.create(
            self.main_control.transform,
            name="roundness",
            default=0,
            min=0,
        )
        upper_roundness_scaled = MultiplyNode.create(f"{self.main_control}_upper_roundness")
        upper_roundness_scaled.input[0].connect_from(self.roundness_attr)
        upper_roundness_scaled.input[1].set(0.5)
        lower_roundness_scaled = MultiplyNode.create(f"{self.main_control}_roundness_invert")
        lower_roundness_scaled.input[0].connect_from(self.roundness_attr)
        lower_roundness_scaled.input[1].set(-0.5)
        roundness_side_offset = MultiplyNode.create(f"{self.main_control}_roundness_side_offset")
        roundness_side_offset.input[0].connect_from(self.roundness_attr)
        roundness_side_offset.input[1].set(-0.25)

        upper_roundness_scaled.output.connect_to(f"{self.upper_control.offset}.translateY")
        lower_roundness_scaled.output.connect_to(f"{self.lower_control.offset}.translateY")
        roundness_side_offset.output.connect_to(f"{self.upper_control.offset}.translateX")
        roundness_side_offset.output.connect_to(f"{self.lower_control.offset}.translateX")

        self.upper_sub_control = create_control(
            f"mouth_corner_{side}_up_sub",
            transform=self.upper_control.transform,
            parent=self.upper_control,
            size=control_size * 0.5,
            direction="z",
        )
        surface_slide_constraint(
            mouth_surface, self.upper_control.transform, self.upper_sub_control.offset
        )
        self.lower_sub_control = create_control(
            f"mouth_corner_{side}_lo_sub",
            transform=self.upper_control.transform,
            parent=self.lower_control,
            size=control_size * 0.5,
            direction="z",
        )
        surface_slide_constraint(
            mouth_surface, self.lower_control.transform, self.lower_sub_control.offset
        )

        self.sub_controls: list[Control] = [
            self.sub_control,
            self.upper_sub_control,
            self.lower_sub_control,
        ]

        if sub_control_vis_attr is not None:
            for control in self.sub_controls:
                sub_control_vis_attr.connect_to(f"{control.transform}.visibility")
