from collections.abc import Sequence

from maya import cmds
from maya.api.OpenMaya import MPoint

from yrig.control import create_control
from yrig.transform import create_transform
from yrig.transform.utils import create_space_switch, get_position


class Look:
    def __init__(
        self,
        part: str = "look",
        side: str = "M",
        parent: str = "face_grp",
        control_parent: str = "neck_M0_head_ctl",
        control_size: float = 1.0,
        parent_jnt: str = "face_jnt",
        parent_spaces: Sequence[str] = (
            "neck_M0_head_ctl",
            "spine_M0_chest_ctl",
            "body_M0_ctl",
            "local_M0_ctl",
        ),
    ):
        self.part: str = part
        self.side: str = side
        self.parent: str = parent
        self.control_parent: str = control_parent
        self.control_size: float = control_size
        self.parent_jnt: str = parent_jnt
        self.guides = ["eye_aim_L", "eye_aim_R"]
        self.parent_list = parent_spaces

    def create_midpoint_locator(
        self,
        transform_a: str,
        transform_b: str,
        name: str = "midpoint_loc",
        world_space: bool = True,
    ) -> str:
        """Create a locator at the midpoint between two transforms.

        Args:
            transform_a: First transform.
            transform_b: Second transform.
            name: Name of the locator to create.
            world_space: Whether to calculate positions in world space.

        Returns:
            The name of the created locator transform.
        """

        pos_a = get_position(transform_a, world_space)
        pos_b = get_position(transform_b, world_space)

        midpoint = MPoint(
            (pos_a.x + pos_b.x) * 0.5,
            (pos_a.y + pos_b.y) * 0.5,
            (pos_a.z + pos_b.z) * 0.5,
        )

        locator = cmds.spaceLocator(name=name)[0]

        cmds.xform(
            locator,  # type:ignore
            worldSpace=world_space,
            translation=[midpoint.x, midpoint.y, midpoint.z],  # type:ignore
        )

        return locator  # type:ignore

    def setup_structure(self) -> None:
        self.main_grp = create_transform(name=f"eye_{self.side}", parent=self.parent)
        self.control_grp = create_transform(name=f"eye_control_{self.side}", parent=self.main_grp)

        """    def create_controls(self) -> None:
        self.main_ctrl = create_control(
            name=self.guides["root_name"],
            parent=self.control_grp,
            transform=self.guides["center_piv"],
            size=self.control_size,
            control_shape="round_square",
            direction="z",
        )"""

    def build(self) -> None:
        self.setup_structure()

        mid_guide: str = self.create_midpoint_locator(
            transform_a=self.guides[0], transform_b=self.guides[1], name="eyelid_mid_loc"
        )

        self.main_ctrl = create_control(
            name="Look_M",
            parent=self.control_grp,
            transform=mid_guide,
            size=self.control_size * 2.5,
            control_shape="round_square",
            direction="z",
            dimensions=(3, 1, 1),
        )
        if self.parent_list:
            create_space_switch(
                target_transform=self.main_ctrl.offset,
                parents=self.parent_list,
                target_control=self.main_ctrl.transform,
            )

        cmds.delete(mid_guide)
