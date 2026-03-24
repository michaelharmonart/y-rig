# type: ignore
"""Component Chain 01 module"""

# import mgear.pymaya as pm

# import maya.cmds as cmds
# import math

from mgear.shifter import component
from mgear.core import primitive, attribute
import mgear.pymaya as pm

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
        self.jnt_pos.append([self.chassis_npo, "chassis", 0, False])

    # =====================================================
    # ATTRIBUTES
    # =====================================================

    def addAttributes(self):
        self.steer_att = self.addAnimParam("steer", "Steer", "double", 0)
        self.frontWheel_spin_att = self.addAnimParam(
            "frontWheelSpin", "Front Wheel Spin", "double", 0
        )
        self.rearWheel_spin_att = self.addAnimParam("rearWheelSpin", "Rear Wheel Spin", "double", 0)
        self.wheelDrive_att = self.addAnimParam("wheelDrive", "Wheel Drive", "double", 0)
        self.steerDrive_att = self.addAnimParam("steerDrive", "Steer Drive", "double", 0)
        self.wheelRadius_att = self.addSetupParam("wheelRadius", "Wheel Radius", "double", 35)
        self.steerRadius_att = self.addSetupParam("steerRadius", "Steer Radius", "double", 35)
        attribute.addProxyAttribute(
            [
                self.steer_att,
                self.frontWheel_spin_att,
                self.rearWheel_spin_att,
                self.wheelDrive_att,
                self.steerDrive_att,
            ],
            [self.drive_ctl],
        )

    # =====================================================
    # OPERATORS
    # =====================================================

    def addOperators(self):
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
        pass

    def connect_wheels(self):
        print("connecting wheels")
        self.connect_standard()

        for comp in self.rig.components:
            if "wheel" not in comp.name:
                continue

            # Steering
            if hasattr(comp, "steer_att"):
                pm.connectAttr(self.steer_att, comp.steer_att, force=True)

            # Drive
            if hasattr(comp, "wheelDrive_att"):
                pm.connectAttr(self.wheelDrive_att, comp.wheelDrive_att, force=True)

            if hasattr(comp, "steerDrive_att"):
                pm.connectAttr(self.steerDrive_att, comp.steerDrive_att, force=True)

            # Radius (setup params)
            if hasattr(comp, "wheelRadius_att"):
                pm.connectAttr(self.wheelRadius_att, comp.wheelRadius_att, force=True)

            if hasattr(comp, "steerRadius_att"):
                pm.connectAttr(self.steerRadius_att, comp.steerRadius_att, force=True)

            # Front vs Rear spin
            if hasattr(comp, "frontWheel_spin_att"):
                if "front" in comp.name.lower():
                    pm.connectAttr(self.frontWheel_spin_att, comp.frontWheel_spin_att, force=True)
                else:
                    pm.connectAttr(self.rearWheel_spin_att, comp.frontWheel_spin_att, force=True)

    def addConnection(self):
        print("adding connections")
        # Guide connector name is 'wheels' in y_car_body_01/guide.py
        self.connections["wheels"] = self.connect_wheels

    # =====================================================
    # CONNECTOR
    # =====================================================

    def setRelation(self):
        self.relatives["root"] = self.root_npo
        self.relatives["chassis"] = self.chassis_npo

        self.controlRelatives["root"] = self.drive_ctl
        self.controlRelatives["chassis"] = self.drive_ctl

        self.jointRelatives["root"] = 0
        self.jointRelatives["chassis"] = 1
