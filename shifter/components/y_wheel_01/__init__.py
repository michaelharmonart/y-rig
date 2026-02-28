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
        t = self.guide.tra["root"]

        # Base group
        self.wheel_npo = primitive.addTransform(self.root, self.getName("wheel_npo"), t)

        # Steering group
        self.steer_grp = primitive.addTransform(self.wheel_npo, self.getName("steer_grp"), t)

        # Spin group
        self.spin_grp = primitive.addTransform(self.steer_grp, self.getName("spin_grp"), t)

        # Control
        self.wheel_ctl = self.addCtl(
            self.spin_grp,
            "wheel_ctl",
            t,
            self.color_fk,
            "circle",
            w=self.size,
            tp=self.parentCtlTag,
        )

        # Deform locator
        self.wheel_loc = primitive.addTransform(
            self.wheel_ctl,
            self.getName("wheel_loc"),
            t,
        )

        self.jnt_pos.append([self.wheel_loc, 0, None, False])

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
            pm.connectAttr(self.spin_att, self.spin_grp.rotateX)
            pm.connectAttr(self.steer_att, self.steer_grp.rotateY)

        # =====================================================
        # CONNECTOR
        # =====================================================

    def setRelation(self):
        self.relatives["root"] = self.wheel_loc
        self.jointRelatives["root"] = 0
        self.controlRelatives["root"] = self.wheel_ctl
