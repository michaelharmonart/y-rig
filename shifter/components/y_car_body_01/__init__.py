# type: ignore
"""Component Chain 01 module"""

# import mgear.pymaya as pm

# import maya.cmds as cmds
# import math

from mgear.shifter import component
from mgear.core import primitive, attribute
import mgear.pymaya as pm
from mgear.core import transform


# from mgear.core import transform


##########################################################
# COMPONENT
##########################################################


class Component(component.Main):
    """Shifter component Class"""

    # =====================================================
    # OBJECTS
    # =====================================================
    frontAxis_bool = True
    backAxis_bool = True

    def addObjects(self):
        t = self.guide.tra["root"]
        t_chassis = self.guide.tra["chassis"]

        self.root_npo = primitive.addTransform(self.root, self.getName("root_npo"), t)

        self.chassis_npo = primitive.addTransform(
            self.root_npo, self.getName("chassis_npo"), t_chassis
        )

        self.body_npo = primitive.addTransform(
            self.chassis_npo, self.getName("body_npo"), t_chassis
        )

        # add controls
        self.drive_ctl = self.addCtl(
            self.root_npo,
            "drive_ctl",
            t_chassis,
            self.color_fk,
            "cube",
            w=self.size * 1.3,
            h=self.size * 0.00002,
            d=self.size * 10.5,
            tp=None,
        )
        pm.parent(self.chassis_npo, self.drive_ctl)

        self.rolloffset = primitive.addTransform(self.root_npo, self.getName("rolloffset"), t)

        pm.parent(self.rolloffset, self.drive_ctl)

        pos = transform.getOffsetPosition(self.rolloffset, [0, 0, 218.403])
        m = transform.setMatrixPosition(self.rolloffset.worldMatrix.get(), pos)

        self.frontAxis_ctrl_OFST = primitive.addTransform(
            self.rolloffset,
            self.getName("frontAxis_ctrl_OFST"),
            m,
        )

        self.frontAxis_ctrl = self.addCtl(
            self.frontAxis_ctrl_OFST,
            "frontAxis_ctrl",
            m,
            self.color_fk,
            "circle",
            w=self.size * 1.5,
            h=self.size * 1.5,
            d=self.size * 1.5,
            tp=None,
        )
        pos2 = transform.getOffsetPosition(self.rolloffset, [0, 0, -137.08])
        m2 = transform.setMatrixPosition(self.rolloffset.worldMatrix.get(), pos2)

        self.rearAxis_ctrl_OFST = primitive.addTransform(
            self.frontAxis_ctrl, self.getName("rearAxis_ctrl_OFST"), m2
        )

        self.rearAxis_ctrl = self.addCtl(
            self.rearAxis_ctrl_OFST,
            "rearAxis_ctrl",
            m2,
            self.color_fk,
            "circle",
            w=self.size * 1.5,
            h=self.size * 1.5,
            d=self.size * 1.5,
            tp=None,
        )

        self.body_ctrl = self.addCtl(
            self.drive_ctl,
            "body_ctrl",
            t_chassis,
            self.color_fk,
            "cube",
            w=self.size * 8.3,
            h=self.size * 0.00002,
            d=self.size * 8.5,
            tp=None,
        )

        pm.parent(self.body_ctrl, self.rearAxis_ctrl)
        pm.parent(self.chassis_npo, self.rearAxis_ctrl)
        # comment

        # add joints
        self.jnt_pos.append([self.root_npo, "root", None, False])
        self.jnt_pos.append([self.chassis_npo, "chassis", 0, False])
        self.jnt_pos.append([self.body_npo, "body", "body", False])

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
        self.steerRadius_att = self.addSetupParam("steerRadius", "Steer Radius", "double", 0)
        pm.addAttr(self.drive_ctl, longName="wheelDrive2", attributeType="double", keyable=True)
        pm.addAttr(self.drive_ctl, longName="wheelRadius2", attributeType="double", keyable=True)
        attribute.addProxyAttribute(
            [
                self.steer_att,
                self.frontWheel_spin_att,
                self.rearWheel_spin_att,
                self.wheelDrive_att,
                self.steerDrive_att,
                self.wheelRadius_att,
                self.steerRadius_att,
            ],
            [self.drive_ctl],
        )

    # =====================================================
    # OPERATORS
    # =====================================================

    def addOperators(self):
        # connect up suspension with the body jnt
        pm.connectAttr(self.body_ctrl.translateY, self.body_npo.translateY, force=True)
        pm.connectAttr(self.body_ctrl.rotateZ, self.body_npo.rotateZ, force=True)
        pm.connectAttr(self.body_ctrl.rotateX, self.body_npo.rotateX, force=True)

        pm.transformLimits(self.body_ctrl, ty=(-5, 5), ety=(True, True))
        pm.transformLimits(self.body_ctrl, rx=(-5, 5), erx=(True, True))
        pm.transformLimits(self.body_ctrl, rz=(-5, 5), erz=(True, True))

        # Lock + hide translates
        pm.setAttr(self.body_ctrl.tx, lock=True, keyable=False, channelBox=False)
        pm.setAttr(self.body_ctrl.tz, lock=True, keyable=False, channelBox=False)

        # Lock + hide rotates
        pm.setAttr(self.body_ctrl.ry, lock=True, keyable=False, channelBox=False)

        # Lock + hide scale (always do this unless needed)
        pm.setAttr(self.body_ctrl.sx, lock=True, keyable=False, channelBox=False)
        pm.setAttr(self.body_ctrl.sy, lock=True, keyable=False, channelBox=False)
        pm.setAttr(self.body_ctrl.sz, lock=True, keyable=False, channelBox=False)

    def connect_wheels(self):
        print("connecting wheels")

        # self.connect_standard()

        # for comp in self.rig.components:
        #     if "wheel" not in comp.name:
        #         continue

        #     # Steering
        #     if hasattr(comp, "steer_att"):
        #         pm.connectAttr(self.steer_att, comp.steer_att, force=True)

        #     # Drive
        #     if hasattr(comp, "wheelDrive_att"):
        #         pm.connectAttr(self.wheelDrive_att, comp.wheelDrive_att, force=True)

        #     if hasattr(comp, "steerDrive_att"):
        #         pm.connectAttr(self.steerDrive_att, comp.steerDrive_att, force=True)

        #     # Radius (setup params)
        #     if hasattr(comp, "wheelRadius_att"):
        #         pm.connectAttr(self.wheelRadius_att, comp.wheelRadius_att, force=True)

        #     if hasattr(comp, "steerRadius_att"):
        #         pm.connectAttr(self.steerRadius_att, comp.steerRadius_att, force=True)

        #     # Front vs Rear spin
        #     if hasattr(comp, "frontWheel_spin_att"):
        #         if "front" in comp.name.lower():
        #             pm.connectAttr(self.frontWheel_spin_att, comp.frontWheel_spin_att, force=True)
        #         else:
        #             pm.connectAttr(self.rearWheel_spin_att, comp.frontWheel_spin_att, force=True)

        #     # disconnect existing connections to steerDrive before connecting
        #     dest_attr = self.drive_ctl.steerDrive

        #     # Check for existing incoming connections
        #     inputs = pm.listConnections(dest_attr, plugs=True, source=True, destination=False)

        #     if inputs:
        #         for src in inputs:
        #             pm.disconnectAttr(src, dest_attr)
        #             print("disconnecting existing connection from {} to {}".format(src, dest_attr))

        #     # Now connect safely
        #     pm.connectAttr(comp.steerDriveDistance_md + ".outputX", dest_attr, force=True)

    def addConnection(self):
        print("adding connections")
        # Guide connector name is 'wheels' in y_car_body_01/guide.py
        self.connections["y_wheel_01"] = self.connect_wheels
        # print("Connector on guide:", self.root.attr("connector").get())
        # print("ending add conntion function")

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


# idea for expression is you have the expression connect to a random node and then connect the nodes output to the drive control!!!
