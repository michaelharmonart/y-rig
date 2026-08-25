from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.transform import create_transform


class Nostril:
    def __init__(
        self,
        side: str,
        guides: dict,
        main_ctrl: str,
        joint_parent: str,
        control_size: float = 1.0,
    ):
        self.side = side
        self.guides = guides
        self.main_ctrl = main_ctrl
        self.joint_parent = joint_parent
        self.control_size = control_size

    def build(self) -> None:

        # =====================================================
        # CENTER CONTROL
        # =====================================================

        self.center_offset = create_transform(
            name=f"nostril_center_{self.side}_offset",
            parent=self.main_ctrl,
            transform=self.guides[f"nostril_center_{self.side}"],
        )

        self.center_ctrl = create_control(
            name=f"nostril_center_{self.side}",
            parent=self.main_ctrl,
            transform=self.guides[f"nostril_center_{self.side}"],
            size=self.control_size * 0.35,
            control_shape="circle",
            direction="z",
        )

        self.center_jnt = create_joint(
            name=f"nostril_center_{self.side}",
            transform=self.center_ctrl,
            parent=self.joint_parent,
        )

        cmds.connectAttr(f"{self.center_ctrl}.translate", f"{self.center_offset}.translate")

        cmds.connectAttr(f"{self.center_ctrl}.rotate", f"{self.center_offset}.rotate")

        # Scale only X/Z
        cmds.connectAttr(
            f"{self.center_ctrl.transform}.scaleX",
            f"{self.center_offset}.scaleX",
        )

        cmds.connectAttr(
            f"{self.center_ctrl.transform}.scaleZ",
            f"{self.center_offset}.scaleZ",
        )

        # =====================================================
        # BACK
        # =====================================================

        self.back_offset = create_transform(
            name=f"nostril_back_{self.side}_offset",
            parent=self.center_ctrl.transform,
            transform=self.guides[f"nostril_back_{self.side}"],
        )

        self.back_ctrl = create_control(
            name=f"nostril_back_{self.side}",
            parent=self.back_offset,
            transform=self.guides[f"nostril_back_{self.side}"],
            size=self.control_size * 0.35,
            control_shape="circle",
            direction="z",
        )

        self.back_jnt_offset = create_transform(
            name=f"nostril_back_{self.side}_jnt_offset",
            parent=self.center_jnt,
            transform=self.guides[f"nostril_back_{self.side}"],
        )

        self.back_jnt = create_joint(
            name=f"nostril_back_{self.side}",
            transform=self.back_ctrl,
            parent=self.back_jnt_offset,
        )

        cmds.connectAttr(
            f"{self.back_ctrl.transform}.translate",
            f"{self.back_jnt_offset}.translate",
        )

        cmds.connectAttr(
            f"{self.back_ctrl.transform}.rotate",
            f"{self.back_jnt_offset}.rotate",
        )

        # =====================================================
        # FRONT
        # =====================================================

        self.front_offset = create_transform(
            name=f"nostril_front_{self.side}_offset",
            parent=self.center_ctrl.transform,
            transform=self.guides[f"nostril_front_{self.side}"],
        )

        self.front_ctrl = create_control(
            name=f"nostril_front_{self.side}",
            parent=self.front_offset,
            transform=self.guides[f"nostril_front_{self.side}"],
            size=self.control_size * 0.35,
            control_shape="circle",
            direction="z",
        )

        self.front_jnt_offset = create_transform(
            name=f"nostril_front_{self.side}_jnt_offset",
            parent=self.center_jnt,
            transform=self.guides[f"nostril_front_{self.side}"],
        )

        self.front_jnt = create_joint(
            name=f"nostril_front_{self.side}",
            transform=self.front_ctrl,
            parent=self.front_jnt_offset,
        )

        cmds.connectAttr(
            f"{self.front_ctrl.transform}.translate",
            f"{self.front_jnt_offset}.translate",
        )

        cmds.connectAttr(
            f"{self.front_ctrl.transform}.rotate",
            f"{self.front_jnt_offset}.rotate",
        )

        # =====================================================
        # BRIDGE
        # =====================================================

        self.bridge_offset = create_transform(
            name=f"nostril_bridge_{self.side}_offset",
            parent=self.center_ctrl.transform,
            transform=self.guides[f"nostril_bridge_{self.side}"],
        )

        self.bridge_ctrl = create_control(
            name=f"nostril_bridge_{self.side}",
            parent=self.bridge_offset,
            transform=self.guides[f"nostril_bridge_{self.side}"],
            size=self.control_size * 0.35,
            control_shape="circle",
            direction="z",
        )

        self.bridge_jnt_offset = create_transform(
            name=f"nostril_bridge_{self.side}_jnt_offset",
            parent=self.center_jnt,
            transform=self.guides[f"nostril_bridge_{self.side}"],
        )

        self.bridge_jnt = create_joint(
            name=f"nostril_bridge_{self.side}",
            transform=self.bridge_ctrl,
            parent=self.bridge_jnt_offset,
        )

        cmds.connectAttr(
            f"{self.bridge_ctrl.transform}.translate",
            f"{self.bridge_jnt_offset}.translate",
        )

        cmds.connectAttr(
            f"{self.bridge_ctrl.transform}.rotate",
            f"{self.bridge_jnt_offset}.rotate",
        )

        # =====================================================
        # OUTSIDE
        # =====================================================

        self.outside_offset = create_transform(
            name=f"nostril_outside_{self.side}_offset",
            parent=self.center_ctrl.transform,
            transform=self.guides[f"nostril_outside_{self.side}"],
        )

        self.outside_ctrl = create_control(
            name=f"nostril_outside_{self.side}",
            parent=self.outside_offset,
            transform=self.guides[f"nostril_outside_{self.side}"],
            size=self.control_size * 0.35,
            control_shape="circle",
            direction="z",
        )

        self.outside_jnt_offset = create_transform(
            name=f"nostril_outside_{self.side}_jnt_offset",
            parent=self.center_jnt,
            transform=self.guides[f"nostril_outside_{self.side}"],
        )

        self.outside_jnt = create_joint(
            name=f"nostril_outside_{self.side}",
            transform=self.outside_ctrl,
            parent=self.outside_jnt_offset,
        )

        cmds.connectAttr(
            f"{self.outside_ctrl.transform}.translate",
            f"{self.outside_jnt_offset}.translate",
        )

        cmds.connectAttr(
            f"{self.outside_ctrl.transform}.rotate",
            f"{self.outside_jnt_offset}.rotate",
        )
