# type: ignore
"""Component y_car__body_01 module"""

# import mgear.pymaya as pm

# import maya.cmds as cmds
# import math

import mgear.pymaya as pm
from mgear.core import attribute, primitive, transform
from mgear.shifter import component

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
        t_left = self.guide.tra["left"]
        t_right = self.guide.tra["right"]
        t_front = self.guide.tra["front"]
        t_back = self.guide.tra["back"]

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
            14,
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
            16,
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
            16,
            "circle",
            w=self.size * 1.5,
            h=self.size * 1.5,
            d=self.size * 1.5,
            tp=None,
        )

        self.body_ctrl = self.addCtl(
            self.rearAxis_ctrl,
            "body_ctrl",
            t_chassis,
            17,
            "cube",
            w=self.size * 8.3,
            h=self.size * 0.00002,
            d=self.size * 8.5,
            tp=None,
        )
        # Ensure the body control Y starts at 0 (neutral suspension)
        pm.setAttr(self.body_ctrl.ty, 0)

        self.root_pivot_npo = primitive.addTransform(
            self.rolloffset, self.getName("root_pivot_npo"), t
        )

        self.left_pivot_npo = primitive.addTransform(
            self.root_pivot_npo, self.getName("left_pivot_npo"), t_left
        )

        self.right_pivot_npo = primitive.addTransform(
            self.left_pivot_npo, self.getName("right_pivot_npo"), t_right
        )

        self.front_pivot_npo = primitive.addTransform(
            self.right_pivot_npo, self.getName("front_pivot_npo"), t_front
        )

        self.back_pivot_npo = primitive.addTransform(
            self.front_pivot_npo, self.getName("back_pivot_npo"), t_back
        )

        self.child_pivot_npo = primitive.addTransform(
            self.back_pivot_npo, self.getName("child_pivot_npo"), t_chassis
        )

        pos2 = transform.getOffsetPosition(self.rolloffset, [0, 200, 0])
        m2 = transform.setMatrixPosition(self.rolloffset.worldMatrix.get(), pos2)

        self.root_pivot_ctrl_GRP = primitive.addTransform(
            self.drive_ctl,
            self.getName("root_pivot_ctrl_GRP"),
            m2,
        )

        self.root_pivot_ctrl = self.addCtl(
            self.root_pivot_ctrl_GRP,
            "root_pivot",
            m2,
            17,
            "circle",
            w=self.size * 1.5,
            h=self.size * 1.5,
            d=self.size * 1.5,
            tp=None,
        )
        pm.parent(self.root_pivot_npo, self.drive_ctl)
        pm.parentConstraint(self.child_pivot_npo, self.rolloffset)

        # pm.parent(self.body_ctrl, self.rearAxis_ctrl)
        pm.parent(self.chassis_npo, self.rearAxis_ctrl)

        # add joints
        self.jnt_pos.append([self.root_npo, "root", None, False])
        self.jnt_pos.append([self.chassis_npo, "chassis", 0, False])
        self.jnt_pos.append([self.body_npo, "body", "chassis", False])
        self.jnt_pos.append([self.root_pivot_npo, "rootPivot", "None", False])
        self.jnt_pos.append([self.left_pivot_npo, "leftPivot", None, False])
        self.jnt_pos.append([self.right_pivot_npo, "rightPivot", None, False])
        self.jnt_pos.append([self.front_pivot_npo, "frontPivot", None, False])
        self.jnt_pos.append([self.back_pivot_npo, "backPivot", None, False])
        self.jnt_pos.append([self.child_pivot_npo, "childPivot", None, False])

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
        self.wheelRadius_att = self.addSetupParam(
            "wheelRadius",
            "Wheel Radius",
            "double",
            self.settings.get("wheelRadius", 35),
        )
        self.wheelRadius2_att = self.addSetupParam(
            "wheelRadius2",
            "Wheel Radius 2",
            "double",
            self.settings.get("wheelRadius2", 35),
        )
        self.steerRadius_att = self.addSetupParam(
            "steerRadius",
            "Steer Radius",
            "double",
            self.settings.get("steerRadius", 0),
        )
        pm.addAttr(self.drive_ctl, longName="wheelDrive2", attributeType="double", keyable=True)
        attribute.addProxyAttribute(
            [
                self.steer_att,
                self.frontWheel_spin_att,
                self.rearWheel_spin_att,
                self.wheelDrive_att,
                self.steerDrive_att,
                self.wheelRadius_att,
                self.wheelRadius2_att,
                self.steerRadius_att,
            ],
            [self.drive_ctl],
        )
        pm.setAttr(self.body_ctrl.ty, 0)

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

        # logic for the pivot joints. root_pibot_ctrl translate X goes into mult divide that is set to input 2 X as -1. the mult divide goes into remap values then into the rotate Z for the right and left pivots. the front and back pivots just get the root_pivot_ctrl translate X directly connected to their rotate Y
        pm.createNode("multiplyDivide", name=self.getName("pivotMD"))
        pm.setAttr(self.getName("pivotMD") + ".input2X", -1)
        pm.connectAttr(
            self.root_pivot_ctrl.translateX, self.getName("pivotMD") + ".input1X", force=True
        )

        # create left side remap value and connect
        pm.createNode("remapValue", name=self.getName("pivotRemap_left"))
        pm.setAttr(self.getName("pivotRemap_left") + ".inputMax", -800)
        pm.setAttr(self.getName("pivotRemap_left") + ".outputMax", -180)
        pm.connectAttr(
            self.getName("pivotMD") + ".outputX",
            self.getName("pivotRemap_left") + ".inputValue",
            force=True,
        )
        pm.connectAttr(
            self.getName("pivotRemap_left") + ".outValue", self.left_pivot_npo.rotateZ, force=True
        )

        # create right side remap value and connect
        pm.createNode("remapValue", name=self.getName("pivotRemap_right"))
        pm.setAttr(self.getName("pivotRemap_right") + ".inputMax", 800)
        pm.setAttr(self.getName("pivotRemap_right") + ".outputMax", 180)
        pm.connectAttr(
            self.getName("pivotMD") + ".outputX",
            self.getName("pivotRemap_right") + ".inputValue",
            force=True,
        )
        pm.connectAttr(
            self.getName("pivotRemap_right") + ".outValue", self.right_pivot_npo.rotateZ, force=True
        )

        # creat front
        pm.createNode("remapValue", name=self.getName("pivotRemap_front"))
        pm.setAttr(self.getName("pivotRemap_front") + ".inputMax", 800)
        pm.setAttr(self.getName("pivotRemap_front") + ".outputMax", 180)
        pm.connectAttr(
            self.root_pivot_ctrl.translateZ,
            self.getName("pivotRemap_front") + ".inputValue",
            force=True,
        )
        pm.connectAttr(
            self.getName("pivotRemap_front") + ".outValue", self.front_pivot_npo.rotateX, force=True
        )

        # create back
        pm.createNode("remapValue", name=self.getName("pivotRemap_back"))
        pm.setAttr(self.getName("pivotRemap_back") + ".inputMax", -800)
        pm.setAttr(self.getName("pivotRemap_back") + ".outputMax", -180)
        pm.connectAttr(
            self.root_pivot_ctrl.translateZ,
            self.getName("pivotRemap_back") + ".inputValue",
            force=True,
        )
        pm.connectAttr(
            self.getName("pivotRemap_back") + ".outValue", self.back_pivot_npo.rotateX, force=True
        )

        # pm.parentConstraint(self.child_pivot_npo, self.rolloffset)

        t_chassis = self.guide.tra["chassis"]
        self.upVector_GRP = primitive.addTransform(
            self.root,
            self.getName("upVector_GRP"),
            t_chassis,
        )

        pm.setAttr(self.body_ctrl.ty, 0)

        """
        add in the ability to move the chassis up and down while adjusting the wheel radius
        """
        pm.createNode("multiplyDivide", name=self.getName("Chassis_y_adjust_MD"))
        pm.setAttr(self.getName("Chassis_y_adjust_MD") + ".input2Y", 1.5)
        pm.connectAttr(
            self.drive_ctl + ".wheelRadius",
            self.getName("Chassis_y_adjust_MD") + ".input1Y",
            force=True,
        )

        pm.createNode("plusMinusAverage", name=self.getName("Chassis_y_adjust_PM"))
        pm.setAttr(self.getName("Chassis_y_adjust_PM") + ".operation", 2)
        pm.connectAttr(
            self.getName("Chassis_y_adjust_MD") + ".outputY",
            self.getName("Chassis_y_adjust_PM") + ".input1D[0]",
            force=True,
        )
        pm.connectAttr(
            self.drive_ctl + ".wheelRadius2",
            self.getName("Chassis_y_adjust_PM") + ".input1D[1]",
            force=True,
        )
        pm.connectAttr(
            self.getName("Chassis_y_adjust_PM") + ".output1D",
            self.chassis_npo + ".translateY",
            force=True,
        )

        self.lock_and_hide_drive_control_attrs()

    def lock_and_hide_drive_control_attrs(self):
        if self.settings.get("lockAndHide", False):  # noqa: SIM102
            if pm.objExists(self.drive_ctl + ".wheelRadius2"):
                pm.setAttr(
                    self.drive_ctl + ".wheelRadius2",
                    lock=True,
                    keyable=False,
                    channelBox=False,
                )
                pm.setAttr(
                    self.drive_ctl + ".wheelRadius",
                    lock=True,
                    keyable=False,
                    channelBox=False,
                )
                # pm.setAttr(
                #     self.drive_ctl + ".steerRadius",
                #     lock=True,
                #     keyable=False,
                #     channelBox=False,
                # )
                pm.setAttr(
                    self.drive_ctl + ".wheelDrive",
                    lock=True,
                    keyable=False,
                    channelBox=False,
                )

    def connect_wheels(self):
        # print("connecting wheels")
        x = 1

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
        # print("adding connections")
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
