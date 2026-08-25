from __future__ import annotations

from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.transform import create_transform


def get_teeth_locator_indices(cvs_count: int) -> list[int]:
    """Return CV indices that should be used as tooth locator anchors.

    The goal is to sample the spline curve at the start, middle, and quarter points
    so the teeth component gets a simple, evenly spaced placement set.
    """

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
    ):
        self.guides = guides
        self.main_ctrl = main_ctrl
        self.joint_parent = joint_parent
        self.control_grp = control_grp
        self.component_grp = component_grp
        self.control_size = control_size

    def build_single_teeth_spline(self, guide_name: str, num: int) -> tuple[object, list[str]]:

        cvs = cmds.ls(f"{guide_name}.cv[*]", flatten=True)
        if not cvs:
            return (), []

        indices = get_teeth_locator_indices(len(cvs))

        cv_controls = []

        group = create_transform(
            name=f"{guide_name}_ofst_grp",
            parent=self.control_grp,
        )
        if num == 1:
            cmds.parentConstraint(
                "jaw_M_ctl",
                group,
                maintainOffset=True,
            )

        main_control = create_control(
            name=f"{guide_name}_main",
            parent=group,
            transform=group,
            size=self.control_size * 0.5,
            control_shape="circle",
            direction="x",
        )

        parent = main_control

        for i, cv_index in enumerate(indices):
            pos = cmds.pointPosition(cvs[cv_index], world=True)
            pos_tuple: tuple[float, float, float] = (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
            )

            offset = create_transform(
                name=f"{guide_name}_{i}_ofs",
                parent=parent,  # type: ignore
            )

            cmds.xform(offset, worldSpace=True, translation=pos_tuple)

            ctrl = create_control(
                name=f"{guide_name}_{i}",
                parent=parent,
                transform=offset,
                size=self.control_size * 0.5,
                control_shape="circle",
                direction="x",
            )

            cv_controls.append(ctrl.transform)

            # parent = ctrl.transform

        spline = matrix_spline_from_transforms(
            name=f"{guide_name}_matrixSpline",
            cv_transforms=cv_controls,
            pinned_transforms=5,
            primary_axis=(1, 0, 0),
            secondary_axis=(0, 0, 1),
            padded=False,
            parent=self.component_grp,
        )

        joint_parent = self.joint_parent

        self.joints = []

        for i, pinned in enumerate(spline.pinned_transforms):
            jnt = create_joint(
                name=f"{guide_name}_{i}",
                parent=joint_parent,
                transform=pinned,
            )

            # Disconnect rotation on the end joints
            if i == 0 or i == len(spline.pinned_transforms) - 1:
                dest = f"{jnt}.rotate"
                source = cmds.connectionInfo(dest, sourceFromDestination=True)

                if source:
                    cmds.disconnectAttr(source, dest)  # type: ignore

            self.joints.append(jnt)

            joint_parent = jnt

        return spline, self.joints

    def cleanup(self) -> None:
        cmds.hide("teeth_M_npo")

    def twist_fix(self, guide: str, all_joints: list) -> None:
        if "bottom" in guide:
            twist_fix_md = cmds.createNode(
                "multiplyDivide",
                name="twist_fix_md",
            )
            cmds.setAttr(f"{twist_fix_md}.input2X", -1)  # type: ignore
            cmds.connectAttr("jaw_M_ctl" + ".rotateX", f"{twist_fix_md}.input1X")

            bottom_joints = all_joints[5:]
            cmds.connectAttr(f"{twist_fix_md}.outputX", f"{bottom_joints[0]}.rotateZ")

    def build_teeth(self) -> list[str]:
        all_joints = []

        for j, guide in enumerate(("top_teeth", "bottom_teeth")):
            if guide not in self.guides:
                continue

            if not cmds.objExists(self.guides[guide]):
                continue

            _, joints = self.build_single_teeth_spline(self.guides[guide], j)
            all_joints.extend(joints)
            self.twist_fix(guide, all_joints)

        self.cleanup()
        return all_joints
