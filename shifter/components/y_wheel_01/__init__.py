# type: ignore
"""Component Chain 01 module"""

import mgear.pymaya as pm
import maya.cmds as cmds

from mgear.shifter import component
from mgear.core import primitive
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
        # Get the position of the guide locators
        t = self.guide.tra["root"]
        t_ball = self.guide.tra["ball"]
        t_steer = self.guide.tra["steer"]
        t_wheel = self.guide.tra["wheel"]
        t_width = self.guide.tra["width"]

        # --- Hierarchy Setup (Transforms) ---

        # Ball pivot (Main parent for both steer and wheel)
        self.root_npo = primitive.addTransform(self.root, self.getName("root_npo"), t)

        self.suspension_y_npo = primitive.addTransform(
            self.root_npo, self.getName("suspension_y_npo"), t_ball
        )

        self.ball_npo = primitive.addTransform(
            self.suspension_y_npo, self.getName("ball_npo"), t_ball
        )

        # Steering offset pivot (Child of ball)
        self.steer_npo = primitive.addTransform(self.ball_npo, self.getName("steer_npo"), t_steer)

        # Wheel spin pivot (Child of ball, NOT child of steer)
        self.wheel_npo = primitive.addTransform(self.ball_npo, self.getName("wheel_npo"), t_wheel)

        # Wheel control hierarchy
        self.frontWheel_reset = primitive.addTransform(
            self.root_npo, self.getName("frontWheel_reset"), t_width
        )

        # Deform locator (final output for binding)
        self.wheel_loc = pm.spaceLocator(n=self.getName("wheel_loc"))[0]

        self.wheel_loc.setMatrix(t_ball)
        pm.parent(self.wheel_loc, self.frontWheel_reset)

        # # Wheel control hierarchy
        # self.frontWheel_reset = primitive.addTransform(
        #     self.root_npo, self.getName("frontWheel_reset"), t_width
        # )

        self.frontWheel_display_ctl = self.addCtl(
            self.frontWheel_reset,
            "frontWheel_display_ctl",
            t_wheel,
            self.color_fk,
            "circle",
            w=self.size * 0.3,
            tp=None,
        )

        self.frontWheel_ctl = self.addCtl(
            self.frontWheel_reset,
            "frontWheel_ctl",
            t_wheel,
            self.color_fk,
            "cube",
            w=self.size * 0.1,
            h=self.size * 0.1,
            d=self.size * 0.1,
            tp=self.frontWheel_display_ctl,
        )

        # Joint outputs
        self.jnt_pos.append([self.ball_npo, "ball", None, False])
        self.jnt_pos.append([self.steer_npo, "steer", "ball", False])
        self.jnt_pos.append([self.wheel_npo, "wheel", "ball", False])

    # =====================================================
    # ATTRIBUTES
    # =====================================================

    def addAttributes(self):
        self.spin_att = self.addSetupParam("spin", "Spin", "double", 0)

        self.steer_att = self.addSetupParam("steer", "Steer", "double", 0)

        self.wheel_radius = self.addSetupParam("wheel_radius", "Wheel Radius", "double", 1)

        """
        ex) michaels arm module
        attribute.addProxyAttribute(self.roll_att, [self.ik_ctl, self.upv_ctl])

        at some point add in a proxy for my controls so that I can still have the wheel module work and use them but then I can connect up the body into the attributes to ccontrol it from there as well.
        """

    # =====================================================
    # OPERATORS
    # =====================================================

    def addOperators(self):
        # Connect animation attributes to transforms
        pm.connectAttr(self.steer_att, self.wheel_loc.rotateY)

        pm.parentConstraint(self.wheel_loc, self.ball_npo, maintainOffset=True, skipTranslate=["y"])
        pm.connectAttr(self.frontWheel_ctl.translateY, self.ball_npo.translateY)

        # wheel radius stuff
        pm.parentConstraint(self.wheel_npo, self.frontWheel_display_ctl, maintainOffset=True)
        # --- Wheel Radius Display Control ---

        shape = self.frontWheel_display_ctl.getShape()

        # get the CVs of the circle
        # --- Wheel Radius Display Control ---

        shape = self.frontWheel_display_ctl.getShape()

        # get the CVs using cmds
        cv_list = cmds.ls(shape.name() + ".cv[*]", fl=True)

        # create multiplyDivide node
        self.radius_md = pm.createNode("multiplyDivide", n=self.getName("wheelRadius_md"))

        # connect wheel radius
        pm.connectAttr(self.wheel_radius, self.radius_md.input1X)

        # default circle size
        pm.setAttr(self.radius_md.input2X, 0.3 * self.size)

        # connect to CV positions
        for cv in cv_list:
            pm.connectAttr(self.radius_md.outputX, cv + ".xValue")
            pm.connectAttr(self.radius_md.outputX, cv + ".zValue")

        # comment here

    # =====================================================
    # CONNECTOR
    # =====================================================

    def setRelation(self):
        self.relatives["root"] = self.ball_npo
        self.controlRelatives["root"] = self.frontWheel_display_ctl

        self.jointRelatives["ball"] = 0
        self.jointRelatives["steer"] = 1
        self.jointRelatives["wheel"] = 2
