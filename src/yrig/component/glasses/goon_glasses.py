from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.transform import create_transform


class Glasses:
    def __init__(
        self,
        part: str = "glasses",
        side: str = "",
        parent: str = "face_grp",
        control_parent: str = "neck_M0_head_ctl",
        control_size: float = 1.0,
        parent_jnt: str = "face_jnt",
    ):
        self.part: str = part
        self.side: str = side
        self.parent: str = parent
        self.control_parent: str = control_parent
        self.control_size: float = control_size
        self.parent_jnt: str = parent_jnt
        self.guides = ["glassesmid", "glassesrim", "glassesarm"]

    # -------------------
    # Build steps
    # -------------------

    def setup_structure(self) -> None:
        self.main_grp = create_transform(name=f"eye_{self.side}", parent=self.parent)
        self.component_grp = create_transform(
            name=f"eye_component_{self.side}", parent=self.main_grp
        )
        cmds.hide(self.component_grp)
        self.control_grp = create_transform(name=f"eye_control_{self.side}", parent=self.main_grp)

    def create_controls(self) -> None:
        self.controls = []
        self.main_ctrl = create_control(
            name="glassesmid_M",
            parent=self.control_grp,
            transform="glassesmid_M",
            size=self.control_size,
            control_shape="round_square",
            direction="z",
            position_offset=(0, 0, 5),
        )
        self.controls.append(self.main_ctrl)

        for side in ["L", "R"]:
            ctrl_parent = self.main_ctrl
            for guide in self.guides:
                ctrl = create_control(
                    name=f"{guide}_{side}",
                    parent=ctrl_parent.transform,
                    transform=f"{guide}_{side}",
                    size=self.control_size,
                    control_shape="circle",
                    direction="x",
                )
                self.controls.append(ctrl)
                ctrl_parent = ctrl

    def create_joints(self) -> None:
        self.joints = []
        jnt_parent = self.parent_jnt
        for control in self.controls:
            if control.name == "glassesmid_R":
                jnt_parent = self.joints[0]
            jnt = create_joint(name=control.name, transform=control.transform, parent=jnt_parent)
            self.joints.append(jnt)
            jnt_parent = jnt

    def build(self) -> None:
        self.setup_structure()
        self.create_controls()
        self.create_joints()
