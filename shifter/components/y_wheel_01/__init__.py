# type: ignore
"""Component Chain 01 module"""

import mgear.pymaya as pm
# from mgear.pymaya import datatypes

from mgear.shifter import component

# from mgear.core import node, applyop, vector
from mgear.core import primitive
# from mgear.core import attribute, transform, primitive

##########################################################
# COMPONENT
##########################################################


class Component(component.Main):
    """Shifter component Class"""

    # =====================================================
    # OBJECTS
    # =====================================================
    def addObjects(self):
        # Get the position of the guide locators and built the component objects based on that
        t = self.guide.tra["root"]

        # --- Hierarchy Setup (Transforms) ---
        # Ball pivot (Main parent for both steer and wheel)
        self.ball_npo = primitive.addTransform(self.root, self.getName("ball_npo"), t)

        # Steering offset pivot (Child of ball)
        self.steer_npo = primitive.addTransform(self.ball_npo, self.getName("steer_npo"), t)

        # Wheel spin pivot (Child of ball, NOT child of steer)
        self.wheel_npo = primitive.addTransform(self.ball_npo, self.getName("wheel_npo"), t)

        # Control
        self.wheel_ctl = self.addCtl(
            self.wheel_npo,
            "wheel_ctl",
            t,
            self.color_fk,
            "circle",
            w=self.size,
            tp=self.parentCtlTag,
        )

        # Deform locator (final output for binding)
        self.wheel_loc = primitive.addTransform(
            self.wheel_ctl,
            self.getName("wheel_loc"),
            t,
        )

        self.jnt_pos.append([self.ball_npo, "ball", None, False])
        self.jnt_pos.append([self.steer_npo, "steer", "ball", False])
        self.jnt_pos.append([self.wheel_loc, "wheel", "ball", False])

    # =====================================================
    # ATTRIBUTES
    # =====================================================
    def addAttributes(self):
        self.spin_att = self.addAnimParam("spin", "Spin", "double", 0)
        self.steer_att = self.addAnimParam("steer", "Steer", "double", 0)

    # =====================================================
    # OPERATORS
    # =====================================================
    def addOperators(self):
        # Connect animation attributes to transforms
        pm.connectAttr(self.spin_att, self.wheel_npo.rotateX)
        pm.connectAttr(self.steer_att, self.ball_npo.rotateY)

    # =====================================================
    # CONNECTOR
    # =====================================================
    def setRelation(self):
        self.relatives["root"] = self.ball_npo
        self.controlRelatives["root"] = self.wheel_ctl

        self.jointRelatives["ball"] = 0
        self.jointRelatives["steer"] = 1
        self.jointRelatives["wheel"] = 2
