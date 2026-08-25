from dataclasses import dataclass

from maya import cmds

from yrig.joint import create_joint
from yrig.maya_api.node import MultiplyDivideNode


@dataclass
class DrivenInfo:
    transform: str
    channels: list[str]
    mults: list[float]


def create_driver_joint(driver: DrivenInfo, driven: DrivenInfo) -> str:
    parent = cmds.listRelatives(driver.transform, parent=True)
    driven_jnt = create_joint(
        name=driven.transform, parent=str(parent[0]), connect=False, transform=driver.transform
    )
    cmds.hide(driven_jnt)
    if len(driver.channels) != len(driven.channels):
        return ""

    for i, attr in enumerate(driver.channels):
        if i % 3 == 0:
            mult_node = MultiplyDivideNode.create(name=f"{driver.transform}_0{i // 3 + 1}_mult")
            multchannel = "X"
        elif i % 3 == 1:
            multchannel = "Y"
        elif i % 3 == 2:
            multchannel = "Z"

        # cmds.connectAttr(attr, driven.channels[i])
        cmds.connectAttr(f"{driver.transform}.{attr}", f"{mult_node.input1}{multchannel}")
        cmds.setAttr(f"{mult_node.input2}{multchannel}", driver.mults[i])  # type:ignore
        cmds.connectAttr(f"{mult_node.output}{multchannel}", f"{driven_jnt}.{driven.channels[i]}")
    return driven_jnt
