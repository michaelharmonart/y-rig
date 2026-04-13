# type: ignore
"""Component Chain 01 module"""

import mgear.pymaya as pm

# import maya.cmds as cmds
import math

from mgear.shifter import component
from mgear.core import primitive
from mgear.core import transform


##########################################################
# COMPONENT
##########################################################


class Component(component.Main):
    """Shifter component Class"""

    # =====================================================
    # OBJECTS
    # =====================================================
    EXPR_NAME = "car_wheelDrive_expr"

    def addObjects(self):
        # Get the position of the guide locators
        t = self.guide.tra["root"]
        t_ball = self.guide.tra["ball"]
        t_steer = self.guide.tra["steer"]
        t_wheel = self.guide.tra["wheel"]
        t_width = self.guide.tra["width"]

        # --- Hierarchy Setup (Transforms) ---

        if self.side == "L":
            txt = f"Creating left wheel: {self.side}"
            print(txt)

        else:
            txt = f"Creating right wheel: {self.side}"
            print(txt)
            t_wheel = transform.setMatrixPosition(
                t_wheel, [-t_wheel.translate.x, t_ball.translate.y, t_wheel.translate.z]
            )
            t_ball = transform.setMatrixPosition(
                t_ball, [-t_ball.translate.x, t_ball.translate.y, t_ball.translate.z]
            )
            t_steer = transform.setMatrixPosition(
                t_steer, [-t_steer.translate.x, t_steer.translate.y, t_steer.translate.z]
            )
            t_width = transform.setMatrixPosition(
                t_width, [-t_width.translate.x, t_width.translate.y, t_width.translate.z]
            )
        # holder = t_wheel
        # t_wheel = t_width
        # t_width = holder

        # Ball pivot (Main parent for both steer and wheel)
        self.root_npo = primitive.addTransform(self.root, self.getName("root_npo"), t)

        self.suspension_y_npo = primitive.addTransform(
            self.root_npo, self.getName("suspension_y_npo"), t_ball
        )

        self.ball_npo = primitive.addTransform(
            self.suspension_y_npo,
            self.getName("ball_npo"),
            t_ball,
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

        # self.drive_ctrl = self.addCtl(
        #     self.frontWheel_reset,
        #     "drive_ctrl",
        #     t_wheel,
        #     self.color_fk,
        #     "circle",
        #     w=self.size * 0.4,
        #     tp=None,
        # )

        # t_display = transform.setMatrixRotation(t_wheel, [90, 0, 0])

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

        self.wheelRadius_att = self.addSetupParam("wheelRadius", "Wheel Radius", "double", 35)

        self.wheelDrive_att = self.addSetupParam("wheelDrive", "Wheel Drive", "double", 0)

        self.steer_att = self.addSetupParam("steer", "Steer", "double", 0)

        self.frontWheel_spin_att = self.addSetupParam(
            "frontWheelSpin", "Front Wheel Spin", "double", 0
        )

        self.steerDrive_att = self.addSetupParam("steerDrive", "Steer Drive", "double", 0)

        self.steerRadius_att = self.addSetupParam("steerRadius", "Steer Radius", "double", 35)

        self.deflate_att = self.addSetupParam("deflate", "Deflate", "double", 0)
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

        pm.connectAttr(self.wheelRadius_att, self.radius_md.input1X)

        pm.setAttr(self.radius_md.input2X, 0.05)

        pm.connectAttr(self.radius_md.outputX, self.frontWheel_display_ctl.scaleX)
        pm.connectAttr(self.radius_md.outputX, self.frontWheel_display_ctl.scaleY)
        pm.connectAttr(self.radius_md.outputX, self.frontWheel_display_ctl.scaleZ)

        # create some varible name that is unique to each wheel and set that as the expression name so the expression will create a new expression for each wheel.

        # expr = """
        # global vector $vPos = << 0, 0, 0 >>;
        # float $distance = 0.0;
        # int $direction = 1;

        # vector $vPosChange = << drive_ctrl.translateX, drive_ctrl.translateY, drive_ctrl.translateZ >>;

        # float $cx = $vPosChange.x - $vPos.x;
        # float $cy = $vPosChange.y - $vPos.y;
        # float $cz = $vPosChange.z - $vPos.z;

        # $distance = sqrt( ($cx * $cx) + ($cy * $cy) + ($cz * $cz) );

        # float $angle = drive_ctrl.rotateY % 360;

        # if ( ($vPosChange.x == $vPos.x) && ($vPosChange.y == $vPos.y) && ($vPosChange.z == $vPos.z) ) {}
        # else {
        #     if ($angle == 0){
        #         if ($vPosChange.z > $vPos.z) $direction = 1;
        #         else $direction = -1;
        #     }

        #     if ( ($angle > 0 && $angle <= 90) || ($angle < -180 && $angle >= -270) ){
        #         if ($vPosChange.x > $vPos.x) $direction *= 1;
        #         else $direction *= -1;
        #     }

        #     if ( ($angle > 90 && $angle <= 180) || ($angle < -90 && $angle >= -180) ){
        #         if ($vPosChange.z > $vPos.z) $direction *= -1;
        #         else $direction *= 1;
        #     }

        #     if ( ($angle > 180 && $angle <= 270) || ($angle < 0 && $angle >= -90) ){
        #         if ($vPosChange.x > $vPos.x) $direction *= -1;
        #         else $direction *= 1;
        #     }

        #     if ( ($angle > 270 && $angle <= 360) || ($angle < -270 && $angle >= -360) ){
        #         if ($vPosChange.z > $vPos.z) $direction *= 1;
        #         else $direction *= -1;
        #     }

        #     drive_ctrl.wheelDrive = drive_ctrl.wheelDrive +
        #         ( $direction * ( ($distance / (6.283185 * drive_ctrl.wheelRadius)) * 360.0 ) );
        # }

        # $vPos = << drive_ctrl.translateX, drive_ctrl.translateY, drive_ctrl.translateZ >>;
        # """
        # driver = self.root.longName()
        # expr = expr.replace("drive_ctrl", driver)

        # # driver = self.drive_ctrl.longName()
        # # expr = expr.replace("drive_ctrl", driver)

        # pm.expression(
        #     name=self.getName("wheelDrive_expr"),
        #     string=expr,
        #     alwaysEvaluate=True,
        # )

        # pm.connectAttr(self.wheelDrive, self.wheel_npo.rotateX)

        # add in the front wheel spin attribute and connect it to the wheel spin pivot
        pm.createNode("plusMinusAverage", n=self.getName("frontWheelSpin_PMA"))
        pm.connectAttr(self.frontWheel_spin_att, self.getName("frontWheelSpin_PMA") + ".input1D[0]")
        pm.connectAttr(self.wheelDrive_att, self.getName("frontWheelSpin_PMA") + ".input1D[1]")
        # pm.connectAttr(self.getName("frontWheelSpin_PMA") + ".output1D", self.wheel_npo.rotateX)

        # connect up front wheel spin up to another PMA so that the right and left wheel spin can be computed separately.
        # make this work when mirroring the component as well, so that the left and right wheel spin can be computed separately.
        # make right side plus minus average set to minus!!!!!!!!
        side = self.side
        if side == "L":
            side = "L"
        else:
            side = "R"

        self.frontWheelSpin_side_PMA = pm.createNode(
            "plusMinusAverage", n=self.getName("frontWheelSpin_" + side + "_PMA")
        )
        pm.connectAttr(
            self.getName("frontWheelSpin_PMA") + ".output1D",
            self.getName("frontWheelSpin_" + side + "_PMA") + ".input1D[0]",
        )
        pm.connectAttr(
            self.steerDrive_att, self.getName("frontWheelSpin_" + side + "_PMA") + ".input1D[1]"
        )
        pm.connectAttr(
            self.getName("frontWheelSpin_" + side + "_PMA") + ".output1D", self.wheel_npo.rotateX
        )

        # make steer control move the wheel with math
        # create a lot of multiply divide nodes to compute the steer drive and then connect that up to the front wheel spin PMA so that it can be added in with the wheel drive to get the final front wheel spin value.
        self.steerCircumferenceCalc_md = pm.createNode(
            "multiplyDivide", n=self.getName("steerCircumferenceCalc_md")
        )
        pm.setAttr(self.steerCircumferenceCalc_md.operation, 1)
        pm.setAttr(self.steerCircumferenceCalc_md.input2X, 2 * math.pi)
        pm.connectAttr(self.steerRadius_att, self.steerCircumferenceCalc_md.input1X)

        self.steerCircumferenceFraction_MD = pm.createNode(
            "multiplyDivide", n=self.getName("steerCircumferenceFraction_md")
        )
        pm.setAttr(self.steerCircumferenceFraction_MD.operation, 2)
        pm.setAttr(self.steerCircumferenceFraction_MD.input2X, 360)
        pm.connectAttr(self.steer_att, self.steerCircumferenceFraction_MD.input1X)

        self.steerDistance_and_invert_MD = pm.createNode(
            "multiplyDivide", n=self.getName("steerDistance_and_invert_md")
        )
        pm.setAttr(self.steerDistance_and_invert_MD.operation, 1)
        pm.setAttr(self.steerDistance_and_invert_MD.input2Y, -1)
        pm.setAttr(self.steerDistance_and_invert_MD.input2Z, -1)
        pm.connectAttr(
            self.steerCircumferenceFraction_MD.outputX, self.steerDistance_and_invert_MD.input1X
        )
        # get the distance the wheel travels based on the steer angle and the wheel radius and then invert that value so that it can be added in with the wheel drive to get the final front wheel spin value.
        pm.connectAttr(
            self.steerCircumferenceCalc_md.outputX, self.steerDistance_and_invert_MD.input2X
        )

        self.wheelCircmferenceCalc_md = pm.createNode(
            "multiplyDivide", n=self.getName("wheelCircumferenceCalc_md")
        )
        pm.setAttr(self.wheelCircmferenceCalc_md.operation, 1)
        pm.setAttr(self.wheelCircmferenceCalc_md.input2X, 2 * math.pi)
        pm.connectAttr(self.wheelRadius_att, self.wheelCircmferenceCalc_md.input1X)

        self.steerDriveCircumferenceFraction_MD = pm.createNode(
            "multiplyDivide", n=self.getName("steerDriveCircumferenceFraction_md")
        )
        pm.setAttr(self.steerDriveCircumferenceFraction_MD.operation, 2)
        pm.connectAttr(
            self.steerDistance_and_invert_MD.outputX,
            self.steerDriveCircumferenceFraction_MD.input1X,
        )
        pm.connectAttr(
            self.wheelCircmferenceCalc_md.outputX, self.steerDriveCircumferenceFraction_MD.input2X
        )

        self.steerDriveDistance_MD = pm.createNode(
            "multiplyDivide", n=self.getName("steerDriveDistance_md")
        )
        pm.setAttr(self.steerDriveDistance_MD.operation, 1)
        pm.setAttr(self.steerDriveDistance_MD.input2X, -360)
        pm.connectAttr(
            self.steerDriveCircumferenceFraction_MD.outputX, self.steerDriveDistance_MD.input1X
        )
        # connect steerDriveDistance to steer radius attribute
        pm.connectAttr(self.steerDriveDistance_MD.outputX, self.steerDrive_att)

        # connect up wheels controls to steer radius.
        # connect it up to steerDistance_and_invert MD so that it can be added in with the steer angle to get the final steer drive value.
        # BUT I NEED TO HAVE THE RIGHT SIDE ONLY HOOK UP TO STEERDISTANCE_AND_INVERT MD INTO THE INPUTY AND THEN THE OUTPUTY GOES INTO THE WHEEL JNT
        pm.connectAttr(self.steerRadius_att, self.wheel_npo.translateX)

        # make the deflate attribute connect to the remap node that willl control the y transtale of the chasis and make the lattice go up
        self.deflate_remap = pm.createNode("remapValue", n=self.getName("deflate_remap"))
        self.deflate_remap.inputMin.set(0)
        self.deflate_remap.inputMax.set(-5)

        self.deflate_remap.outputMin.set(0)
        self.deflate_remap.outputMax.set(15)

        pm.connectAttr(self.deflate_att, self.deflate_remap.inputValue)
        # CONNECT UP REMAP OUTVALUE TO THE LATTICE GROUP TRANSLATEY ADN CONNECT THE DEFLATE ATTR TO THE CHASIS JOINT (BUT NOT REALLY THE CHASIS JOINT THE SECOND ROOT JOINT)

    # =====================================================
    # CONNECTION
    # =====================================================

    def addConnection(self):
        """Overloads the standard connection so wheel can wire parent body attrs."""
        self.connections["y_car_body_01"] = self.connect_wheel_to_parent

    def connect_wheel_to_parent(self):
        print("connecting wheel to parent")
        self.connect_standard()

        parent = getattr(self, "parent_comp", None)
        if not parent:
            return
        print("found parent")

        # -----------------------------------
        # FIND DRIVER
        # -----------------------------------
        if hasattr(parent, "drive_ctl"):
            driver = parent.drive_ctl
        else:
            print("No drive_ctl found")
            return

        # -----------------------------------
        # CREATE EXPRESSION ONLY ONCE
        # -----------------------------------
        if not pm.objExists(self.EXPR_NAME):
            print("Creating global wheel expression")

            expr = """
            global vector $vPos = << 0, 0, 0 >>;
            float $distance = 0.0;
            int $direction = 1;

            vector $vPosChange = << drive_ctrl.translateX, drive_ctrl.translateY, drive_ctrl.translateZ >>;

            float $cx = $vPosChange.x - $vPos.x;
            float $cy = $vPosChange.y - $vPos.y;
            float $cz = $vPosChange.z - $vPos.z;

            $distance = sqrt( ($cx * $cx) + ($cy * $cy) + ($cz * $cz) );

            float $angle = drive_ctrl.rotateY % 360;

            if ( ($vPosChange.x == $vPos.x) && ($vPosChange.y == $vPos.y) && ($vPosChange.z == $vPos.z) ) {}
            else {
                if ($angle == 0){
                    if ($vPosChange.z > $vPos.z) $direction = 1;
                    else $direction = -1;
                }

                if ( ($angle > 0 && $angle <= 90) || ($angle < -180 && $angle >= -270) ){
                    if ($vPosChange.x > $vPos.x) $direction *= 1;
                    else $direction *= -1;
                }

                if ( ($angle > 90 && $angle <= 180) || ($angle < -90 && $angle >= -180) ){
                    if ($vPosChange.z > $vPos.z) $direction *= -1;
                    else $direction *= 1;
                }

                if ( ($angle > 180 && $angle <= 270) || ($angle < 0 && $angle >= -90) ){
                    if ($vPosChange.x > $vPos.x) $direction *= -1;
                    else $direction *= 1;
                }

                if ( ($angle > 270 && $angle <= 360) || ($angle < -270 && $angle >= -360) ){
                    if ($vPosChange.z > $vPos.z) $direction *= 1;
                    else $direction *= -1;
                }

                drive_ctrl.wheelDrive = drive_ctrl.wheelDrive +
                    ( $direction * ( ($distance / (6.283185 * drive_ctrl.wheelRadius)) * 360.0 ) );
            }

            $vPos = << drive_ctrl.translateX, drive_ctrl.translateY, drive_ctrl.translateZ >>;
            """

            # expr = expr.replace("DRIVER", driver.longName())
            # expr = expr.replace("driveNode", driver.longName())

            driver = driver.longName()
            expr = expr.replace("drive_ctrl", driver)
            expr = expr.replace("wheelDrive", "wheelDrive2")

            pm.expression(
                name=self.EXPR_NAME,
                string=expr,
                alwaysEvaluate=True,
            )

        else:
            print("Expression already exists, skipping")

        print("connected drive and steer drive from parent")

        # Match parent car body attributes if present
        if hasattr(parent, "steer_att") and hasattr(self, "steer_att"):
            pm.connectAttr(parent.steer_att, self.steer_att, force=True)

        # if hasattr(parent, "wheelDrive_att") and hasattr(self, "wheelDrive_att"):
        #     pm.connectAttr(parent.wheelDrive_att, self.wheelDrive_att, force=True)

        if hasattr(parent, "steerDrive_att") and hasattr(self, "steerDrive_att"):
            pm.connectAttr(parent.steerDrive_att, self.steerDrive_att, force=True)

        if hasattr(parent, "wheelRadius_att") and hasattr(self, "wheelRadius_att"):
            pm.connectAttr(parent.wheelRadius_att, self.wheelRadius_att, force=True)

        if hasattr(parent, "steerRadius_att") and hasattr(self, "steerRadius_att"):
            pm.connectAttr(parent.steerRadius_att, self.steerRadius_att, force=True)

        # Gently support the 4-wheel front/rear spin split
        if (
            hasattr(parent, "frontWheel_spin_att")
            and hasattr(parent, "rearWheel_spin_att")
            and hasattr(self, "frontWheel_spin_att")
        ):
            if "front" in self.name.lower() or "front" in self.fullName.lower():
                pm.connectAttr(parent.frontWheel_spin_att, self.frontWheel_spin_att, force=True)
            else:
                pm.connectAttr(parent.rearWheel_spin_att, self.frontWheel_spin_att, force=True)

        # Connect wheel's computed steer drive into the body
        if hasattr(parent, "steerDrive_att"):
            pm.connectAttr(self.steerDriveDistance_MD.outputX, parent.steerDrive_att, force=True)

        # if pm.isConnected("global_C0_ctl.message", "car_body_C0_drive_ctl.uiHost_cnx"):
        #     pm.disconnectAttr("global_C0_ctl.message", "car_body_C0_drive_ctl.uiHost_cnx")

        # connect up the expression with the drive control
        # pm.connectAttr(self.parent.drive_ctl.wheelDrive, self.wheelDrive_att, force=True)
        # pm.connectAttr(
        #     self.parent.drive_ctl.steerDrive, self.frontWheelSpin_side_PMA + ".input1D", force=True
        # )

        # if hasattr(parent, "drive_ctl"):
        #     pm.parentConstraint(parent.drive_ctl, self.root, maintainOffset=True)
        #     print("parent constrained wheel root to drive_ctl")
        # else:
        #     print("parent has no drive_ctl")

        # # connect drive control translate z into wheel root translate z
        # if hasattr(parent, "drive_ctl"):
        #     print("attemptintg to connect drive_ctl translateZ to wheel root translateZ")
        #     pm.connectAttr(parent.drive_ctl.translateZ, self.ball_npo.translateZ, force=True)
        #     print("connected drive_ctl translateZ to wheel root translateZ")
        # else:
        #     print("parent has no drive_ctl to connect translateZ")

        # connect drive ctl to front wheel spin PNA input 1D
        if hasattr(parent, "drive_ctl"):
            print("going to try and connect up frontWheelSpin_PMA")
            pm.connectAttr(
                parent.drive_ctl.wheelDrive2,
                self.getName("frontWheelSpin_PMA") + ".input1D[1]",
                force=True,
            )
            print("connected drive_ctl wheelDrive to frontWheelSpin_PMA input1D[1]")
        else:
            print("parent has no drive_ctl to connect wheelDrive")

        # chagne the steer radius to be based off the wheel guide and not hard coded in car_body_01
        if self.side == "R":
            print("adjusting steer radius for right side")
            pm.createNode("multiplyDivide", n=self.getName("steerRadius_invert_md"))

            pm.setAttr(
                self.getName("steerRadius_invert_md") + ".input2X", -1
            )  # Invert the steer radius
            pm.connectAttr(
                parent.steerRadius_att,
                self.getName("steerRadius_invert_md") + ".input1X",
                force=True,
            )
            pm.connectAttr(
                self.getName("steerRadius_invert_md") + ".outputX",
                self.wheel_npo.translateX,
                force=True,
            )

        wheel_pos = self.guide.pos["wheel"]
        ball_pos = self.guide.pos["ball"]

        steer_radius = wheel_pos.x - ball_pos.x

        parent.drive_ctl.steerRadius.set(steer_radius)

        print("finished connecting wheel to parent")

    def setRelation(self):
        self.relatives["root"] = self.ball_npo
        self.controlRelatives["root"] = self.frontWheel_display_ctl

        self.jointRelatives["ball"] = 0
        self.jointRelatives["steer"] = 1
        self.jointRelatives["wheel"] = 2


# don't let drive control override the wheel root steer drive so attribute is still connected.
# next connect drive control translate z into wheel root translate z
# then connecte drive ctl wheel drive into the frontWheelSpin_R_PMA
