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

        # ---------------------------------------------------------
        # TONGUE GUIDES
        # ---------------------------------------------------------

        guide_order = [
            "tongue_back",
            "tongue2",
            "tongue3",
            "tongue4",
            "tongue5",
            "tongue_front",
        ]

        # ---------------------------------------------------------
        # CREATE CONTROLS AND JOINTS
        #
        # Controls are all direct children of control_grp.
        #
        # Joints remain parented in a chain.
        # ---------------------------------------------------------

        self.controls = []
        self.joints = []

        joint_parent = self.joint_parent if cmds.objExists(self.joint_parent) else None

        for guide in guide_order:
            # -----------------------------------------------------
            # CONTROL
            # -----------------------------------------------------

            ctrl = create_control(
                name=f"{guide}_ctrl",
                parent=self.control_grp,
                transform=self.guides[guide],
                size=self.control_size * 0.5,
                control_shape="circle",
                direction="x",
            )

            ctrl_transform = ctrl.transform if hasattr(ctrl, "transform") else ctrl

            self.controls.append(ctrl)

            # -----------------------------------------------------
            # JOINT
            # -----------------------------------------------------

            joint = create_joint(
                name=guide,
                parent=joint_parent,
                transform=ctrl_transform,
                connect=False,
            )

            self.joints.append(joint)

            # Next joint is parented to this joint.
            joint_parent = joint

            # -----------------------------------------------------
            # CONTROL -> JOINT
            # -----------------------------------------------------

            cmds.parentConstraint(
                ctrl_transform,
                joint,
                maintainOffset=False,
            )

        # ---------------------------------------------------------
        # ORIENT JOINT CHAIN
        # ---------------------------------------------------------

        if self.joints:
            cmds.joint(
                self.joints[0],
                edit=True,
                orientJoint="xyz",
                secondaryAxisOrient="yup",
                children=True,
                zeroScaleOrient=True,
            )

        # ---------------------------------------------------------
        # CLEANUP
        # ---------------------------------------------------------

        self.curve = None
        self.ik_handle = None