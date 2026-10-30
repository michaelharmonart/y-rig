from typing import Literal

from maya import cmds

from yrig.control import Control, create_control
from yrig.maya_api.attribute import BooleanAttribute, ScalarAttribute
from yrig.maya_api.node import MultiplyNode
from yrig.surface import surface_slide_constraint
from yrig.transform import create_transform
from yrig.transform.utils import connect_transform


class MouthCorner:
    def __init__(
        self,
        side: Literal["L", "R"],
        guide: str,
        mouth_surface: str,
        mouth_surface_local: str,
        control_parent: Control | str,
        parent: str,
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
        self.main_local_npo = create_transform(
            f"mouth_corner_{side}_local_npo", transform=self.main_control.transform, parent=parent
        )
        self.main_local = create_transform(f"mouth_corner_{side}_local", parent=self.main_local_npo)
        connect_transform(self.main_control.transform, self.main_local)

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

        self.sub_local_npo = create_transform(
            f"mouth_corner_{side}_sub_local_npo",
            transform=guide,
            parent=self.main_local,
        )
        self.sub_local = create_transform(
            f"mouth_corner_{side}_sub_local",
            parent=self.sub_local_npo,
        )
        connect_transform(self.sub_control.transform, self.sub_local)

        surface_slide_constraint(
            mouth_surface_local,
            driver_transform=self.main_local,
            slider_transform=self.sub_local_npo,
        )

        self.upper_control = create_control(
            f"mouth_corner_{side}_up",
            transform=self.sub_local_npo,
            parent=self.sub_control,
            size=control_size * 0.5,
            direction="z",
        )
        self.upper_local_npo = create_transform(
            f"mouth_corner_{side}_up_local_npo",
            transform=self.sub_local_npo,
            parent=self.sub_local_npo,
        )
        self.upper_local = create_transform(
            f"mouth_corner_{side}_up_local",
            parent=self.upper_local_npo,
        )
        connect_transform(self.upper_control.transform, self.upper_local)

        self.lower_control = create_control(
            f"mouth_corner_{side}_lo",
            transform=self.sub_local_npo,
            parent=self.sub_control,
            size=control_size * 0.5,
            direction="z",
        )
        self.lower_local_npo = create_transform(
            f"mouth_corner_{side}_lo_local_npo",
            transform=self.sub_local_npo,
            parent=self.sub_local_npo,
        )
        self.lower_local = create_transform(
            f"mouth_corner_{side}_lo_local",
            parent=self.lower_local_npo,
        )
        connect_transform(self.lower_control.transform, self.lower_local)

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
        upper_roundness_scaled.output.connect_to(f"{self.upper_local_npo}.translateY")
        lower_roundness_scaled.output.connect_to(f"{self.lower_control.offset}.translateY")
        lower_roundness_scaled.output.connect_to(f"{self.lower_local_npo}.translateY")
        roundness_side_offset.output.connect_to(f"{self.upper_control.offset}.translateX")
        roundness_side_offset.output.connect_to(f"{self.upper_local_npo}.translateX")
        roundness_side_offset.output.connect_to(f"{self.lower_control.offset}.translateX")
        roundness_side_offset.output.connect_to(f"{self.lower_local}.translateX")

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

        self.upper_sub_local_npo = create_transform(
            f"mouth_corner_{side}_up_sub_local_npo",
            parent=self.upper_local,
        )
        self.upper_sub_local = create_transform(
            f"mouth_corner_{side}_up_sub_local",
            parent=self.upper_sub_local_npo,
        )
        connect_transform(self.upper_sub_control.transform, self.upper_sub_local)
        surface_slide_constraint(mouth_surface_local, self.upper_local, self.upper_sub_local_npo)

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

        self.lower_sub_local_npo = create_transform(
            f"mouth_corner_{side}lo_sub_local_npo",
            parent=self.lower_local,
        )
        self.lower_sub_local = create_transform(
            f"mouth_corner_{side}lo_sub_local",
            parent=self.lower_sub_local_npo,
        )
        connect_transform(self.lower_sub_control.transform, self.lower_sub_local)
        surface_slide_constraint(mouth_surface_local, self.lower_local, self.lower_sub_local_npo)

        self.sub_controls: list[Control] = [
            self.sub_control,
            self.upper_sub_control,
            self.lower_sub_control,
        ]

        if sub_control_vis_attr is not None:
            for control in self.sub_controls:
                sub_control_vis_attr.connect_to(f"{control.transform}.visibility")
