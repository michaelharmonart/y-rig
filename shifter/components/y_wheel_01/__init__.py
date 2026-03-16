# type: ignore
"""Component Chain 01 module"""

import mgear.pymaya as pm
# import maya.cmds as cmds

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
        # self.spin_att = self.addSetupParam("spin", "Spin", "double", 0)

        # self.steer_att = self.addSetupParam("steer", "Steer", "double", 0)

        self.wheelRadius = self.addSetupParam("wheelRadius", "Wheel Radius", "double", 1)

        self.wheelDrive = self.addSetupParam("wheelDrive", "Wheel Drive", "double", 0)

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

        # get the CVs of the circle
        # --- Wheel Radius Display Control ---

        self.radius_md = pm.createNode("multiplyDivide", n=self.getName("wheelRadius_md"))

        pm.connectAttr(self.wheelRadius, self.radius_md.input1X)

        pm.setAttr(self.radius_md.input2X, 0.05)

        pm.connectAttr(self.radius_md.outputX, self.frontWheel_display_ctl.scaleX)
        pm.connectAttr(self.radius_md.outputX, self.frontWheel_display_ctl.scaleY)
        pm.connectAttr(self.radius_md.outputX, self.frontWheel_display_ctl.scaleZ)

        expr = """
        global vector $vPos = << 0, 0, 0 >>;
        float $distance = 0.0;
        int $direction = 1;
        vector $vPosChange = `getAttr drive_ctrl.translate`;
        float $cx = $vPosChange.x - $vPos.x;
        float $cy = $vPosChange.y - $vPos.y;
        float $cz = $vPosChange.z - $vPos.z;
        $distance = sqrt( `pow $cx 2` + `pow $cy 2` + `pow $cz 2` );
        float $angle = drive_ctrl.rotateY%360;

        if ( ( $vPosChange.x == $vPos.x ) && ( $vPosChange.y == $vPos.y ) && ( $vPosChange.z == $vPos.z ) ){}
        else {
            if ( $angle == 0 ){ 
                if ( $vPosChange.z > $vPos.z ) $direction = 1;
                else $direction=-1;}
            if ( ( $angle > 0 && $angle <= 90 ) || ( $angle <- 180 && $angle >= -270 ) ){ 
                if ( $vPosChange.x > $vPos.x ) $direction = 1 * $direction;
                else $direction = -1 * $direction; }
            if ( ( $angle > 90 && $angle <= 180 ) || ( $angle < -90 && $angle >= -180 ) ){
                if ( $vPosChange.z > $vPos.z ) $direction = -1 * $direction;
                else $direction = 1 * $direction; }
            if ( ( $angle > 180 && $angle <= 270 ) || ( $angle < 0 && $angle >= -90 ) ){
                if ( $vPosChange.x > $vPos.x ) $direction = -1 * $direction;
                else $direction = 1 * $direction; }
            if ( ( $angle > 270 && $angle <= 360 ) || ( $angle < -270 && $angle >= -360 ) ) {
                if ( $vPosChange.z > $vPos.z ) $direction = 1 * $direction;
                else $direction = -1 * $direction; }

            drive_ctrl.wheelDrive = drive_ctrl.wheelDrive + ( ( $direction * ( ( $distance / ( 6.283185 * drive_ctrl.wheelRadius ) ) * 360.0 ) ) ); 
        }

        $vPos = << drive_ctrl.translateX, drive_ctrl.translateY, drive_ctrl.translateZ >>;
        """
        driver = self.root.name()
        expr = expr.replace("drive_ctrl", driver)

        pm.expression(
            name=self.getName("wheelDrive_expr"),
            string=expr,
            alwaysEvaluate=True,
        )

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
