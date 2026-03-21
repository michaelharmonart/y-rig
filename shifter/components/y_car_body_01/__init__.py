# type: ignore
"""Component Chain 01 module"""

# import mgear.pymaya as pm

# import maya.cmds as cmds
# import math

from mgear.shifter import component
from mgear.core import primitive, attribute
# from mgear.core import transform


##########################################################
# COMPONENT
##########################################################


class Component(component.Main):
    """Shifter component Class"""

    # =====================================================
    # OBJECTS
    # =====================================================

    def addObjects(self):
        t = self.guide.tra["root"]
        t_chassis = self.guide.tra["chassis"]

        self.root_npo = primitive.addTransform(self.root, self.getName("root_npo"), t)

        self.chassis_npo = primitive.addTransform(
            self.root_npo, self.getName("chassis_npo"), t_chassis
        )

        # add controls
        self.drive_ctl = self.addCtl(
            self.chassis_npo,
            "drive_ctl",
            t_chassis,
            self.color_fk,
            "cube",
            w=self.size * 0.3,
            h=self.size * 0.2,
            d=self.size * 0.5,
            tp=None,
        )

        # add joints
        self.jnt_pos.append([self.root_npo, "root", None, False])
        self.jnt_pos.append([self.chassis_npo, "chassis", "root", False])

    # =====================================================
    # ATTRIBUTES
    # =====================================================

    def addAttributes(self):
        self.steer_att = attribute.addAttribute(self.drive_ctl, "steer", "double", 0)

        self.frontWheel_spin_att = attribute.addAttribute(
            self.drive_ctl, "frontWheelSpin", "double", 0
        )

        self.rearWheel_spin_att = attribute.addAttribute(
            self.drive_ctl, "rearWheelSpin", "double", 0
        )

        self.wheelDrive_att = attribute.addAttribute(self.drive_ctl, "wheelDrive", "double", 0)

        self.steerDrive_att = attribute.addAttribute(self.drive_ctl, "steerDrive", "double", 0)

        self.wheelRadius_att = attribute.addAttribute(self.drive_ctl, "wheelRadius", "double", 35)

        self.steerRadius_att = attribute.addAttribute(self.drive_ctl, "steerRadius", "double", 35)

    # =====================================================
    # OPERATORS
    # =====================================================

    def addOperators(self):
        pass
        # for comp in self.parent.components:
        #     if "wheel" in comp.name:

        #         pm.connectAttr(self.steer_att, comp.steer_att)
        #         pm.connectAttr(self.wheelDrive_att, comp.wheelDrive)
        #         pm.connectAttr(self.steerDrive_att, comp.steerDrive_att)
        #         pm.connectAttr(self.wheelRadius_att, comp.wheelRadius)
        #         pm.connectAttr(self.steerRadius_att, comp.SteerRadius)

        #         # front vs rear logic
        #         if "front" in comp.name:
        #             pm.connectAttr(self.frontWheel_spin_att, comp.frontWheel_spin_att)
        #         else:
        #             pm.connectAttr(self.rearWheel_spin_att, comp.frontWheel_spin_att)

    # =====================================================
    # CONNECTOR
    # =====================================================

    def setRelation(self):
        self.relatives["root"] = self.root_npo
        self.relatives["chassis"] = self.chassis_npo
