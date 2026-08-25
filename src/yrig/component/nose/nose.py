from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.transform import create_transform

from .nostril import Nostril
from .tip import NoseTip


class Nose:
    def __init__(
        self,
        part: str = "nose",
        side: str = "M",
        parent: str = "face_grp",
        control_parent: str = "neck_M0_head_ctl",
        control_size: float = 1.0,
        parent_jnt: str = "face_jnt",
    ):
        self.part = part
        self.side = side
        self.parent = parent
        self.control_parent = control_parent
        self.control_size = control_size
        self.parent_jnt = parent_jnt

        self.guides: dict[str, str] = {
            "root": "nose_root_M",
            "bridge": "nose_bridge_M",
            "tip": "nose_tip_M",
            "nose_L": "nose_L",
            "nose_R": "nose_R",
            "nostril_center_L": "nostril_center_L",
            "nostril_back_L": "nostril_back_L",
            "nostril_front_L": "nostril_front_L",
            "nostril_bridge_L": "nostril_bridge_L",
            "nostril_outside_L": "nostril_outside_L",
            "nostril_center_R": "nostril_center_R",
            "nostril_back_R": "nostril_back_R",
            "nostril_front_R": "nostril_front_R",
            "nostril_bridge_R": "nostril_bridge_R",
            "nostril_outside_R": "nostril_outside_R",
        }

    # Structure

    def setup_structure(self) -> None:

        self.main_grp = create_transform(
            name=f"nose_{self.side}",
            parent=self.parent,
        )

        self.component_grp = create_transform(
            name=f"nose_component_{self.side}",
            parent=self.main_grp,
        )

        cmds.hide(self.component_grp)

        self.control_grp = create_transform(
            name=f"nose_control_{self.side}",
            parent=self.main_grp,
        )

    def create_controls(self) -> None:

        # print(self.guides["root"])
        # print(cmds.objExists(self.guides["root"]))

        self.main_ctrl = create_control(
            name="nose_M",
            parent=self.control_grp,
            transform=self.guides["bridge"],
            size=self.control_size,
            control_shape="round_square",
            direction="z",
        )

    def create_joints(self) -> None:

        self.main_jnt = create_joint(
            name="root",
            parent=self.parent_jnt,
            transform=self.main_ctrl.transform,
        )

    # Build

    def build(self) -> None:

        self.setup_structure()
        self.create_controls()
        self.create_joints()

        self.tip = NoseTip(
            guides=self.guides,
            main_ctrl=self.main_ctrl.transform,
            joint_parent=self.main_jnt,
            control_grp=self.control_grp,
            component_grp=self.component_grp,
            control_size=self.control_size,
        )

        self.tip.build_tip()

        self.nostril_l = Nostril(
            side="L",
            guides=self.guides,
            main_ctrl=self.tip.tip_ctrl.transform,
            joint_parent=self.tip.left_jnt,
            control_size=self.control_size,
        )

        self.nostril_l.build()

        self.nostril_r = Nostril(
            side="R",
            guides=self.guides,
            main_ctrl=self.tip.tip_ctrl.transform,
            joint_parent=self.tip.right_jnt,
            control_size=self.control_size,
        )

        self.nostril_r.build()

        # cmds.parent("nostril_center_R_jnt", "nose_R_jnt")
        # cmds.parent("nostril_center_L_jnt", "nose_L_jnt")

        # cmds.parent("nostril_center_L_npo", "nose_tip_M_ctl")
        # cmds.parent("nostril_center_R_npo", "nose_tip_M_ctl")
