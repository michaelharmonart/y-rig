from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import mgear.pymaya as pm

if TYPE_CHECKING:
    from ..y_limb_01 import Component as LimbComponent  # relative import within the package
else:
    from y_limb_01 import Component as LimbComponent  # runtime: mGear adds parent dir to sys.path


#############################################
# COMPONENT
#############################################


class Component(LimbComponent):
    """Shifter component Class"""

    GUIDE_MAP: ClassVar[dict[str, str]] = {"mid": "elbow", "end": "wrist"}

    # =====================================================
    # OBJECTS
    # =====================================================
    def addObjects(self) -> None:
        """Add all the objects needed to create the component."""
        super().addObjects()

    # =====================================================
    # ATTRIBUTES
    # =====================================================
    def addAttributes(self) -> None:
        """Create the anim and setupr rig attributes for the component"""
        super().addAttributes()

    # =====================================================
    # OPERATORS
    # =====================================================
    def addOperators(self) -> None:
        """Create operators and set the relations for the component rig

        Apply operators, constraints, expressions to the hierarchy.
        In order to keep the code clean and easier to debug,
        we shouldn't create any new object in this method.

        """
        super().addOperators()

    # =====================================================
    # CONNECTOR
    # =====================================================

    def setRelation(self) -> None:
        """Set the relation beetween object from guide to rig"""
        super().setRelation()
        self.aliasRelatives["eff"] = "hand"

    def addConnection(self) -> None:
        """Add more connection definition to the set"""

        self.connections["shoulder_01"] = self.connect_shoulder_01

    def connect_standard(self) -> None:
        """standard connection definition for the component"""
        super().connect_standard()

    def connect_shoulder_01(self) -> None:
        """Custom connection to be use with shoulder 01 component"""
        self.connect_standard()
        pm.parent(self.rollRef[0], self.ikHandleUpvRef, self.parent_comp.ctl)  # type: ignore

    def finalize(self) -> None:
        """Tag split joints for automatic weight splitting.

        Uses the jnt_pos indices recorded during addObjects to look up
        the actual joints from self.jointList (populated by jointStructure).
        Each segment (upper arm / forearm) is tagged independently.
        """
        super().finalize()
        # test commit
