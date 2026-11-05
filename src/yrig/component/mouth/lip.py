from __future__ import annotations

from dataclasses import dataclass

from yrig.control import Control, create_control
from yrig.joint import create_joint
from yrig.surface import surface_slide_constraint
from yrig.transform import create_transform, matrix_constraint
from yrig.transform.constraint import local_constraint
from yrig.transform.utils import connect_transform


def _default_lip_guides(side: str) -> LipGuides:
    return LipGuides(
        lip_mid_left=f"{side}_lip_mid_L",
        lip_mid=f"{side}_lip_mid_M",
        lip_mid_right=f"{side}_lip_mid_R",
    )


@dataclass
class LipGuides:
    lip_mid_left: str
    lip_mid: str
    lip_mid_right: str


class Lip:
    def __init__(
        self,
        upper: bool,
        guides: LipGuides,
        mouth_surface: str,
        mouth_surface_local: str,
        parent: str,
        joint_parent: str,
        control_parent: Control | str,
        control_follow: Control | str,
        mouth_slide: str,
        mouth_slide_ref: str,
        control_size: float = 1,
    ):
        self.guides = guides
        side_string = "up" if upper else "lo"
        self.name = f"{side_string}_lip"
        self.group = create_transform(self.name, parent=parent)

        self.lip_follow_space = create_transform(
            f"{self.name}_follow_space", parent=str(control_parent)
        )
        matrix_constraint(str(control_follow), self.lip_follow_space, keep_offset=False)
        self.lip_follow = create_transform(f"{self.name}_follow", self.lip_follow_space)
        local_constraint(mouth_slide, self.lip_follow, reference_space=mouth_slide_ref)

        self.lip_move_npo = create_transform(
            f"{self.name}_move_npo", transform=self.guides.lip_mid, parent=str(control_parent)
        )
        matrix_constraint(self.lip_follow, self.lip_move_npo)
        self.lip_move = create_transform(f"{self.name}_move", parent=self.lip_move_npo)
        self.lip_move_local_npo = create_transform(
            f"{self.name}_move_local_npo", transform=self.lip_move, parent=parent
        )
        self.lip_move_local = create_transform(
            f"{self.name}_move_local", parent=self.lip_move_local_npo
        )
        connect_transform(self.lip_move, self.lip_move_local)

        self.slider = create_transform(f"{self.name}_slide", parent=str(control_parent))
        self.slider_local_npo = create_transform(
            f"{self.name}_slide_local_npo", transform=self.slider, parent=parent
        )
        self.slider_local = create_transform(
            f"{self.name}_slide_local", transform=self.slider, parent=self.slider_local_npo
        )

        surface_slide_constraint(
            mouth_surface, driver_transform=self.lip_move, slider_transform=self.slider
        )

        surface_slide_constraint(
            mouth_surface_local,
            driver_transform=self.lip_move_local,
            slider_transform=self.slider_local,
        )

        self.mid_control = create_control(
            f"{self.name}_M",
            transform=guides.lip_mid,
            parent=self.slider,
            size=control_size,
            direction="z",
        )
        self.mid_local_npo = create_transform(
            f"{self.name}_M_local_npo",
            transform=guides.lip_mid,
            parent=self.slider_local,
        )
        self.mid_local = create_transform(
            f"{self.name}_M_local",
            parent=self.mid_local_npo,
        )
        connect_transform(self.mid_control.transform, self.mid_local)

        self.mid_joint = create_joint(f"{self.name}_M", self.mid_local, parent=joint_parent)
