from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.skin.split.tag import tag_for_weight_split
from yrig.transform import create_transform

from .build_teeth import TeethSpline


class Teeth:
    def __init__(
        self,
        part: str = "teeth",
        side: str = "M",
        parent: str = "face_grp",
        control_parent: str = "neck_M0_head_ctl",
        control_size: float = 1.0,
        parent_jnt: str = "face_jnt",
        big_teeth: bool = True,
    ):
        self.part = part
        self.side = side
        self.parent = parent
        self.control_parent = control_parent
        self.control_size = control_size
        self.parent_jnt = parent_jnt
        self.big_teeth = big_teeth

        self.guides: dict[str, str] = {
            "root": "jaw_M",
            "top_teeth": "teeth_top_M",
            "bottom_teeth": "teeth_bottom_M",
        }

    def setup_structure(self) -> None:

        self.main_grp = create_transform(
            name=f"teeth_{self.side}",
            parent=self.parent,
        )

        self.component_grp = create_transform(
            name=f"teeth_component_{self.side}",
            parent=self.main_grp,
        )

        cmds.hide(self.component_grp)

        self.control_grp = create_transform(
            name=f"teeth_control_{self.side}",
            parent=self.control_parent,
        )

    def create_controls(self) -> None:

        self.main_ctrl = create_control(
            name=f"teeth_{self.side}",
            parent=self.control_grp,
            transform=self.guides["root"],
            size=self.control_size,
            control_shape="round_square",
            direction="z",
        )

    def create_joints(self) -> None:

        cmds.select(cl=True)  # type: ignore

        parent = self.parent_jnt if cmds.objExists(self.parent_jnt) else None

        if parent:
            cmds.select(parent)

        source_transform = None

        if hasattr(self.main_ctrl, "transform"):
            source_transform = self.main_ctrl.transform

        self.main_jnt = create_joint(
            name=f"{self.part}_{self.side}_root",
            parent=parent,
            transform=source_transform,
        )

    def build(self) -> None:

        self.setup_structure()
        self.create_controls()
        self.create_joints()

        teeth_builder = TeethSpline(
            guides=self.guides,
            main_ctrl=(
                self.main_ctrl.transform
                if hasattr(self.main_ctrl, "transform")
                else self.guides["root"]
            ),
            joint_parent=self.main_jnt,
            control_grp=self.control_grp,
            component_grp=self.component_grp,
            control_size=self.control_size,
            big_teeth=self.big_teeth,
        )

        top_joints, bottom_joints = teeth_builder.build_teeth()

        self.teeth_joints = top_joints + bottom_joints

        if top_joints:
            tag_for_weight_split(
                influence=top_joints[0],
                split_influences=top_joints,
            )

        if bottom_joints:
            tag_for_weight_split(
                influence=bottom_joints[0],
                split_influences=bottom_joints,
            )
