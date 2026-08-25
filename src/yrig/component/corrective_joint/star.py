from dataclasses import dataclass

from maya import cmds

from yrig.joint import create_joint
from yrig.maya_api.node import EulerToQuatNode, QuatSlerpNode, QuatToEulerNode, RemapValueNode


@dataclass
class PopInfo:
    axis: str
    pop_mult: float
    pop_clampup: float
    pop_clampdown: float


def build_star_joint(
    name: str,
    joint: str,
    ignore_axes: list[str],
    pop_axes: list[PopInfo],
    twist_axis: str = "X",
    local_distance: float = 3,
) -> None:
    root_joints = []
    bind_joints = []

    joint_parent = cmds.listRelatives(joint, parent=True)[0]
    star_jnt = create_joint(
        name=f"{name}_main", transform=joint, connect=False, parent=joint_parent
    )
    blend_axes = [axis for axis in ["X", "Y", "Z"] if axis != twist_axis]

    for axis in ["X", "Y", "Z"]:
        cmds.connectAttr(f"{joint}.translate{axis}", f"{star_jnt}.translate{axis}")
        cmds.connectAttr(f"{joint}.scale{axis}", f"{star_jnt}.scale{axis}")

    etq = EulerToQuatNode.create(name=f"{name}_etq")
    etq.input_rotate.connect_from(f"{joint}.rotate")
    etq.input_rotate_order.connect_from(f"{joint}.rotateOrder")

    qslerp = QuatSlerpNode.create(name=f"{name}_qslerp")
    qslerp.input1_quat.connect_from(etq.output_quat)
    qslerp.input_t.set(0.5)
    cmds.setAttr(f"{qslerp}.angleInterpolation", 1)  # type:ignore

    qte = QuatToEulerNode.create(name=f"{name}_etq")
    qte.input_quat.connect_from(qslerp.output_quat)
    qte.input_rotate_order.connect_from(f"{joint}.rotateOrder")
    qte.output_rotate.connect_to(f"{star_jnt}.rotate")

    for axis in blend_axes:
        if axis not in ignore_axes:
            rootjnt = create_joint(
                name=f"{name}_{axis}_root", transform=joint, connect=False, parent=star_jnt
            )
            cmds.setAttr(f"{rootjnt}.translate{axis}", local_distance)  # type:ignore
            jnt = create_joint(
                name=f"{name}_{axis}", transform=rootjnt, connect=False, parent=rootjnt
            )
            root_joints.append(rootjnt)
            bind_joints.append(jnt)
        if f"-{axis}" not in ignore_axes:
            rootjnt = create_joint(
                name=f"{name}_n{axis}_root", transform=joint, connect=False, parent=star_jnt
            )
            cmds.setAttr(f"{rootjnt}.translate{axis}", -local_distance)  # type:ignore
            jnt = create_joint(
                name=f"{name}_n{axis}", transform=rootjnt, connect=False, parent=rootjnt
            )
            root_joints.append(rootjnt)
            bind_joints.append(jnt)

    for pop in pop_axes:
        temp_axis = pop.axis.lstrip("-")
        if temp_axis != pop.axis:
            mod = "n"
        else:
            mod = ""
        swing_axis = [axis for axis in blend_axes if axis != temp_axis]
        pop_remap = RemapValueNode.create(name=f"{name}_{pop.axis}_remap")
        pop_remap.input_value.connect_from(f"{joint}.rotate{swing_axis[0]}")
        pop_remap.input_max.set(pop.pop_clampup)
        pop_remap.output_max.set(pop.pop_clampup * pop.pop_mult)
        pop_remap.input_min.set(pop.pop_clampdown)
        pop_remap.output_min.set(pop.pop_clampdown * pop.pop_mult)
        pop_remap.output.connect_to(f"{name}_{mod}{temp_axis}_jnt.translate{temp_axis}")
