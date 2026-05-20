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
    # frontAxis_bool = True

    def addObjects(self):
        # Get the position of the guide locators
        t = self.guide.tra["root"]
        t_ball = self.guide.tra["ball"]
        t_steer = self.guide.tra["steer"]
        t_wheel = self.guide.tra["wheel"]
        t_width = self.guide.tra["width"]
        t_lower_arm = self.guide.tra["lower_arm"]
        t_lower_ball = self.guide.tra["lower_ball"]
        t_upper_arm = self.guide.tra["upper_arm"]
        t_front_arm = self.guide.tra["front_arm"]
        t_upperSpring = self.guide.tra["upperSpring"]
        t_lowerSpring = self.guide.tra["lowerSpring"]

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
            t_lower_arm = transform.setMatrixPosition(
                t_lower_arm,
                [-t_lower_arm.translate.x, t_lower_arm.translate.y, t_lower_arm.translate.z],
            )
            t_lower_ball = transform.setMatrixPosition(
                t_lower_ball,
                [
                    -t_lower_ball.translate.x,
                    t_lower_ball.translate.y,
                    t_lower_ball.translate.z,
                ],
            )
            t_upper_arm = transform.setMatrixPosition(
                t_upper_arm,
                [
                    -t_upper_arm.translate.x,
                    t_upper_arm.translate.y,
                    t_upper_arm.translate.z,
                ],
            )
            t_front_arm = transform.setMatrixPosition(
                t_front_arm,
                [
                    -t_front_arm.translate.x,
                    t_front_arm.translate.y,
                    t_front_arm.translate.z,
                ],
            )
            t_upperSpring = transform.setMatrixPosition(
                t_upperSpring,
                [
                    -t_upperSpring.translate.x,
                    t_upperSpring.translate.y,
                    t_upperSpring.translate.z,
                ],
            )
            t_lowerSpring = transform.setMatrixPosition(
                t_lowerSpring,
                [
                    -t_lowerSpring.translate.x,
                    t_lowerSpring.translate.y,
                    t_lowerSpring.translate.z,
                ],
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
        wheel_type = self.settings.get("wheelType", 0)
        print("wheel_type value:", wheel_type)
        if wheel_type == 0:
            print("front wheel")
        else:
            print("rear wheel")

        if wheel_type == 0:
            self.steer_npo = primitive.addTransform(
                self.ball_npo, self.getName("steer_npo"), t_steer
            )

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

        self.frontWheel_display_ctl.rotateZ.set(90)

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

        """
        creating all suspension joints
        """

        width_side = "front" if wheel_type == 0 else "rear"
        self.width_npo = primitive.addTransform(self.ball_npo, self.getName("width_npo"), t_width)

        self.lower_arm_npo = primitive.addTransform(
            self.width_npo, self.getName(f"{width_side}_lower_arm_npo"), t_lower_arm
        )
        self.lower_ball_npo = primitive.addTransform(
            self.lower_arm_npo, self.getName(f"{width_side}_lower_ball_npo"), t_lower_ball
        )

        self.upperarm_npo = primitive.addTransform(
            self.width_npo, self.getName(f"{width_side}_upper_arm_npo"), t_upper_arm
        )

        self.frontArm_npo = primitive.addTransform(
            self.width_npo, self.getName(f"{width_side}_front_arm_npo"), t_front_arm
        )

        # make the upper and lower spring npos have z facing out and y facing out and x facing up

        self.upperSpring_npo = primitive.addTransform(
            self.upperarm_npo, self.getName("frontSpring_npo"), t_upperSpring
        )

        self.lowerSpring_npo = primitive.addTransform(
            self.lower_ball_npo, self.getName("lowerSpring_npo"), t_lowerSpring
        )

        # Joint outputs
        self.jnt_pos.append([self.ball_npo, "ball", None, False])
        if wheel_type == 0:
            self.jnt_pos.append([self.steer_npo, "steer", "ball", False])
        self.jnt_pos.append([self.wheel_npo, "wheel", "ball", False])

        self.jnt_pos.append([self.width_npo, "width", "ball", False])
        self.jointRelatives["width"] = len(self.jnt_pos) - 1

        self.jnt_pos.append([self.lower_arm_npo, f"{width_side}_lower_arm", "width", False])
        self.jointRelatives[f"{width_side}_lower_arm"] = len(self.jnt_pos) - 1
        self.jnt_pos.append(
            [self.lower_ball_npo, f"{width_side}_lower_ball", f"{width_side}_lower_arm", False]
        )
        self.jointRelatives[f"{width_side}_lower_ball"] = len(self.jnt_pos) - 1

        self.jnt_pos.append([self.upperarm_npo, f"{width_side}_upper_arm", "width", False])
        self.jointRelatives[f"{width_side}_upper_arm"] = len(self.jnt_pos) - 1
        self.jnt_pos.append([self.frontArm_npo, f"{width_side}_front_arm", "width", False])
        self.jnt_pos.append(
            [self.upperSpring_npo, f"{width_side}_upperSpring", f"{width_side}_upper_arm", False]
        )
        self.jnt_pos.append(
            [self.lowerSpring_npo, f"{width_side}_lowerSpring", f"{width_side}_lower_ball", False]
        )

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
        if side == "L":
            pm.setAttr(self.frontWheelSpin_side_PMA.operation, 1)

        else:
            pm.setAttr(self.frontWheelSpin_side_PMA.operation, 2)
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

        """
        connect up and do all the suspension logic for the under part like springs n stuff
        """
        pm.connectAttr(self.frontArm_npo.rotateZ, self.upperarm_npo.rotateZ, force=True)
        pm.connectAttr(self.frontArm_npo.rotateZ, self.lower_arm_npo.rotateZ, force=True)

        pm.createNode("multiplyDivide", n=self.getName("frontArm_invert_md"))
        pm.setAttr(self.getName("frontArm_invert_md") + ".input2X", -1)
        pm.connectAttr(
            self.frontArm_npo.rotateZ, self.getName("frontArm_invert_md") + ".input1X", force=True
        )
        pm.connectAttr(
            self.getName("frontArm_invert_md") + ".outputX", self.lower_ball_npo.rotateZ, force=True
        )

        t_front_arm = self.guide.tra["front_arm"]
        if self.side == "R":
            t_front_arm = transform.setMatrixPosition(
                t_front_arm,
                [-t_front_arm.translate.x, t_front_arm.translate.y, t_front_arm.translate.z],
            )

        # create locator directly above the front arm and parent contraint it to the width npo

        self.frontArm_locator = pm.spaceLocator(n=self.getName("frontArm_locator"))[0]
        t_front_arm_locator = transform.setMatrixPosition(
            t_front_arm, [t_front_arm.translate.x, 80, t_front_arm.translate.z]
        )
        self.frontArm_locator.setMatrix(t_front_arm_locator)
        pm.parentConstraint(self.width_npo, self.frontArm_locator, maintainOffset=True)

        # create aim constaints  with aim vector x at 1 and world up type as object up and the upvector is the locator we just created and only do the z axis and its with out ball joint and our front arm joint
        axis = 1
        if self.side == "R":
            axis = -1
        pm.aimConstraint(
            self.ball_npo,
            self.frontArm_npo,
            aimVector=[axis, 0, 0],
            worldUpType="object",
            worldUpObject=self.frontArm_locator,
            skip=["x", "y"],
        )

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

        def reparent_width_joint():
            width_jnt = pm.PyNode(self.getName("width_jnt"))

            body_jnt = pm.PyNode(parent.getName("body_jnt"))

            pm.parent(width_jnt, body_jnt)

        pm.evalDeferred(reparent_width_joint)

        if hasattr(parent, "body_npo"):
            pm.parent(self.width_npo, parent.body_npo, absolute=True)
            # pm.parent(self.lower_arm_npo, parent.body_npo, absolute=True)
            # pm.parent(self.lower_ball_npo, parent.body_npo, absolute=True)
            # pm.parent(self.upperarm_npo, parent.body_npo, absolute=True)
            # pm.parent(self.frontArm_npo, parent.body_npo, absolute=True)
            print("reparented all suspension NPOs under parent.body_npo")
        else:
            print("parent has no body_npo; suspension remains under ball_npo")

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
        wheel_type = self.settings.get("wheelType", 0)

        if wheel_type == 0:
            if hasattr(parent, "steer_att") and hasattr(self, "steer_att"):
                pm.connectAttr(parent.steer_att, self.steer_att, force=True)

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
            if wheel_type == 0:
                pm.connectAttr(parent.frontWheel_spin_att, self.frontWheel_spin_att, force=True)
            else:
                pm.connectAttr(parent.rearWheel_spin_att, self.frontWheel_spin_att, force=True)

        # Connect wheel's computed steer drive into the body
        if wheel_type == 0:
            print(wheel_type)
            print("connecting front wheel steer drive to parent")
            if hasattr(parent, "steerDrive_att"):
                print("connecting wheel steer drive to parent")
                pm.connectAttr(
                    self.steerDriveDistance_MD.outputX, parent.steerDrive_att, force=True
                )

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

        # front and back axis control position changed based off of wheel position. Position should be based off of the wheel position and not hard coded in the car body so that it can be adjusted for different wheel positions and sizes.
        if wheel_type == 0:
            if parent.frontAxis_bool:
                print("adjusting front axis control position based on wheel position")
                pm.parent(parent.frontAxis_ctrl_OFST, world=True)
                pm.parent(parent.rearAxis_ctrl_OFST, world=True)
                parent.frontAxis_bool = False
                ball_z2 = self.guide.pos["ball"].z
                parent.frontAxis_ctrl_OFST.translateZ.set(ball_z2)
                print("setting front axis control position based on ball position:", ball_z2)
                print("re-parenting front and rear axis controls to root")
                pm.parent(parent.frontAxis_ctrl_OFST, parent.rolloffset)
                pm.parent(parent.rearAxis_ctrl_OFST, parent.frontAxis_ctrl)

            else:
                print("front axis control position already set, skipping")

        if wheel_type == 1:
            if parent.backAxis_bool:
                print("adjusting back axis control position based on wheel position")

                parent.backAxis_bool = False

                rear_ws_matrix = parent.rearAxis_ctrl_OFST.getMatrix(worldSpace=True)

                pm.parent(parent.rearAxis_ctrl_OFST, world=True)
                pm.parent(parent.body_ctrl, world=True)
                pm.parent(parent.chassis_npo, world=True)

                parent.rearAxis_ctrl_OFST.setMatrix(rear_ws_matrix, worldSpace=True)

                ball_z2 = self.guide.pos["ball"].z

                ws_pos = pm.xform(parent.rearAxis_ctrl_OFST, q=True, ws=True, t=True)

                pm.xform(parent.rearAxis_ctrl_OFST, ws=True, t=[ws_pos[0], ws_pos[1], ball_z2])

                pm.parent(parent.rearAxis_ctrl_OFST, parent.frontAxis_ctrl, absolute=True)
                pm.parent(parent.body_ctrl, parent.rearAxis_ctrl, absolute=True)
                pm.parent(parent.chassis_npo, parent.rearAxis_ctrl, absolute=True)

                print("rear axis adjusted")

        # parent the upVector_grp with the frontArm Locators
        pm.parent(self.frontArm_locator, parent.upVector_GRP, absolute=True)

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
