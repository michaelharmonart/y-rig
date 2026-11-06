from __future__ import annotations

from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.transform import create_transform, matrix_constraint


def get_teeth_locator_indices(cvs_count: int) -> list[int]:
    """Return CV indices used for tooth control placement."""

    if cvs_count <= 1:
        return [0]

    if cvs_count == 2:
        return [0, 1]

    indices = [0]

    for offset in (0.25, 0.5, 0.75):
        index = round((cvs_count - 1) * offset)

        if index not in indices:
            indices.append(index)

    if indices[-1] != cvs_count - 1:
        indices.append(cvs_count - 1)

    return sorted(indices)


class TeethSpline:
    def __init__(
        self,
        guides: dict,
        main_ctrl: str,
        joint_parent: str,
        control_grp: str,
        component_grp: str,
        control_size: float = 1.0,
        big_teeth: bool = True,
    ):
        self.guides = guides
        self.main_ctrl = main_ctrl
        self.joint_parent = joint_parent
        self.control_grp = control_grp
        self.component_grp = component_grp
        self.control_size = control_size
        self.big_teeth = big_teeth
        self.teeth_controls = []

    def setup_single_tooth_proxies(self) -> None:
        """Create one shared singleTooth attribute across all teeth controls."""

        master_attr = f"{self.main_ctrl}.singleTooth"

        # Create the master attribute if it doesn't exist.
        if not cmds.attributeQuery(
            "singleTooth",
            node=self.main_ctrl,
            exists=True,
        ):
            cmds.addAttr(
                self.main_ctrl,
                longName="singleTooth",
                niceName="Single Tooth",
                attributeType="double",
                minValue=0,
                maxValue=1,
                defaultValue=0,
                keyable=True,
            )

        # Create proxy attributes on every teeth control.
        for control in self.teeth_controls:
            if control == self.main_ctrl:
                continue

            if cmds.attributeQuery(
                "singleTooth",
                node=control,
                exists=True,
            ):
                continue

            cmds.addAttr(
                control,
                longName="singleTooth",
                proxy=master_attr,
            )

    def build_single_teeth_spline(
        self,
        guide_name: str,
        num: int,
    ) -> list[str]:

        cvs = cmds.ls(
            f"{guide_name}.cv[*]",
            flatten=True,
        )

        if not cvs:
            return []

        indices = get_teeth_locator_indices(len(cvs))

        group = create_transform(
            name=f"{guide_name}_ofst_grp",
            parent=self.control_grp,
        )

        if num == 0:
            matrix_constraint("face_mid_M_driven", group)

        if num == 1:
            matrix_constraint("jaw_M_ctl", group)

        main_control = create_control(
            name=f"{guide_name}_main",
            parent=group,
            transform=group,
            size=self.control_size * 0.5,
            control_shape="circle",
            direction="x",
        )

        self.teeth_controls.append(main_control.transform)

        parent = main_control.transform
        joint_parent = self.joint_parent

        joints = []

        for i, cv_index in enumerate(indices):
            pos = cmds.pointPosition(
                cvs[cv_index],
                world=True,
            )

            pos_tuple = (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
            )

            offset = create_transform(
                name=f"{guide_name}_{i}_ofs",
                parent=parent,
            )

            cmds.xform(
                offset,
                worldSpace=True,
                translation=pos_tuple,
            )

            ctrl = create_control(
                name=f"{guide_name}_{i}",
                parent=parent,
                transform=offset,
                size=self.control_size * 0.5,
                control_shape="circle",
                direction="x",
            )

            self.teeth_controls.append(ctrl.transform)

            jnt = create_joint(
                name=f"{guide_name}_{i}",
                parent=joint_parent,
                transform=ctrl.transform,
            )

            joints.append(jnt)

            joint_parent = jnt
            # parent = ctrl.transform

        # Set the main control pivot to the middle tooth joint.
        if joints:
            middle_joint = joints[len(joints) // 2]

            pivot = cmds.xform(
                middle_joint,
                query=True,
                worldSpace=True,
                translation=True,
            )

            pivot = (
                float(pivot[0]),  # ty: ignore[not-subscriptable]
                float(pivot[1]),  # ty: ignore[not-subscriptable]
                float(pivot[2]),  # ty: ignore[not-subscriptable]
            )

            cmds.xform(
                main_control.transform,
                worldSpace=True,
                rotatePivot=pivot,
                scalePivot=pivot,
            )

        return joints

    def cleanup(self) -> None:
        if cmds.objExists("teeth_M_npo"):
            cmds.hide("teeth_M_npo")

    def setup_teeth_visibility(self) -> None:
        master_attr = f"{self.main_ctrl}.singleTooth"

        reverse = cmds.createNode(
            "reverse",
            name="teeth_single_tooth_visibility_REV",
        )

        # Reverse the master attribute.
        cmds.connectAttr(
            master_attr,
            f"{reverse}.inputX",
            force=True,
        )

        # Regular teeth: visible when singleTooth = 0.
        if cmds.objExists("teeth_geo"):
            cmds.connectAttr(
                f"{reverse}.outputX",
                "teeth_geo.visibility",
                force=True,
            )

        # Big teeth: visible when singleTooth = 1.
        if cmds.objExists("big_tooth_geo"):
            cmds.connectAttr(
                master_attr,
                "big_tooth_geo.visibility",
                force=True,
            )

    def build_teeth(self) -> tuple[list[str], list[str]]:
        top_joints = []
        bottom_joints = []

        if cmds.objExists(self.guides["top_teeth"]):
            top_joints = self.build_single_teeth_spline(
                self.guides["top_teeth"],
                0,
            )

        if cmds.objExists(self.guides["bottom_teeth"]):
            bottom_joints = self.build_single_teeth_spline(
                self.guides["bottom_teeth"],
                1,
            )

        # Add the shared attribute and proxies after all controls exist.
        self.setup_single_tooth_proxies()
        if self.big_teeth:
            self.setup_teeth_visibility()
        self.cleanup()

        return top_joints, bottom_joints
