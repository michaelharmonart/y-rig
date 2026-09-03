from __future__ import annotations

from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint


class TongueSpine:
    def __init__(
        self,
        guides: dict,
        main_ctrl: str,
        joint_parent: str,
        control_grp: str,
        component_grp: str,
        control_size: float = 1.0,
    ):
        self.guides = guides
        self.main_ctrl = main_ctrl
        self.joint_parent = joint_parent
        self.control_grp = control_grp
        self.component_grp = component_grp
        self.control_size = control_size

    def build_tongue_spine(self) -> None:

        guide_order = [
            "tongue_back",
            "tongue1",
            "tongue2",
            "tongue_front",
        ]

        self.controls = []
        self.joints = []

        joint_parent = self.joint_parent if cmds.objExists(self.joint_parent) else None

        for guide in guide_order:
            # CONTROL

            ctrl = create_control(
                name=f"{guide}_ctrl",
                parent=self.main_ctrl,
                transform=self.guides[guide],
                size=self.control_size * 1,
                control_shape="circle",
                direction="z",
            )

            ctrl_transform = ctrl.transform

            self.controls.append(ctrl)

            # JOINT

            joint = create_joint(
                name=guide,
                parent=joint_parent,
                transform=ctrl,
                connect=True,
            )

            self.joints.append(joint)

            # Next joint is parented to this joint.
            joint_parent = joint

        # CLEANUP

        self.curve = None
        self.ik_handle = None
