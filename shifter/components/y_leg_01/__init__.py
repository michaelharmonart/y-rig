# type: ignore
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import mgear.pymaya as pm
from mgear.core import attribute, primitive, transform

if TYPE_CHECKING:
    from ..y_limb_01 import Component as LimbComponent  # relative import within the package
else:
    from y_limb_01 import Component as LimbComponent  # runtime: mGear adds parent dir to sys.path

#############################################
# COMPONENT
#############################################


class Component(LimbComponent):
    """Shifter component Class"""

    GUIDE_MAP: ClassVar[dict[str, str]] = {"mid": "knee", "end": "ankle"}
    WORLD_ALIGN_IK = True

    # =====================================================
    # OBJECTS
    # =====================================================
    def addObjects(self) -> None:
        """Add all the objects needed to create the component."""
        super().addObjects()

    def addAttributes(self) -> None:
        super().addAttributes()

    def _add_reference_array_attributes(self) -> None:
        # Ref
        if self.settings["ikrefarray"]:
            ref_names = self.get_valid_alias_list(self.settings["ikrefarray"].split(","))
            if len(ref_names) > 1:
                self.ikref_att = self.addAnimEnumParam("ikref", "Ik Ref", 0, ref_names)

        ref_names = ["Auto", "ikFoot", "World_ctl"]
        if self.settings["upvrefarray"]:
            ref_names += self.get_valid_alias_list(self.settings["upvrefarray"].split(","))
        if len(ref_names) > 1:
            self.upvref_att = self.addAnimEnumParam("upvref", "UpV Ref", 0, ref_names)

        if self.settings["pinrefarray"]:
            ref_names = self.get_valid_alias_list(self.settings["pinrefarray"].split(","))
            ref_names = ["Auto"] + ref_names
            if len(ref_names) > 1:
                self.pin_att = self.addAnimEnumParam("midref", "Mid Control Space", 0, ref_names)

        if self.validProxyChannels:
            attribute.addProxyAttribute(
                [self.blend_att, self.roundness_att],
                [
                    self.fk0_ctl,
                    self.fk1_ctl,
                    self.fk2_ctl,
                    self.ik_ctl,
                    self.upv_ctl,
                ],
            )
            attribute.addProxyAttribute(self.roll_att, [self.ik_ctl, self.upv_ctl])

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

    def _setup_ik_upv(self) -> None:
        # 1 bone chain Upv ref ==============================
        self.ikHandleUpvRef = primitive.addIkHandle(
            self.root,
            self.getName("ikHandleLimbChainUpvRef"),
            self.limbChainUpvRef,
            "ikSCsolver",
        )
        pm.pointConstraint(self.ik_ctl, self.ikHandleUpvRef)
        # handle special case for full mirror behaviour negating
        # scaleY axis to -1
        if self.upv_cns.sy.get() < 0:
            references = []
            for x in [self.limbChainUpvRef[0], self.ik_ctl]:
                ref_trans_name = self.upv_cns.getName() + "_" + x.getName() + "_space_ref"
                ref_trans = primitive.addTransform(
                    x,
                    ref_trans_name,
                )
                transform.matchWorldTransform(self.upv_cns, ref_trans)
                references.append(ref_trans)
            pm.parentConstraint(references[0], references[1], self.upv_cns, mo=True)
        else:
            pm.parentConstraint(self.limbChainUpvRef[0], self.ik_ctl, self.upv_cns, mo=True)

    # =====================================================
    # CONNECTOR
    # =====================================================
    def setRelation(self) -> None:
        """Set the relation beetween object from guide to rig"""
        super().setRelation()
        self.aliasRelatives["eff"] = "foot"

    def connect_standard(self) -> None:
        super().connect_standard()

    def _connect_reference_array(self) -> None:
        # Set the Ik Reference
        self.connectRef(self.settings["ikrefarray"], self.ik_cns)
        if self.settings["upvrefarray"]:
            self.connectRef("Auto,Foot," + self.settings["upvrefarray"], self.upv_cns, True)
        else:
            self.connectRef("Auto,Foot", self.upv_cns, True)

        if self.settings["pinrefarray"]:
            self.connectRef2(
                "Auto," + self.settings["pinrefarray"],
                self.mid_cns,
                self.pin_att,
                [self.ctrn_loc],
                False,
            )

    def finalize(self) -> None:
        """Tag split joints for automatic weight splitting.

        Uses the jnt_pos indices recorded during addObjects to look up
        the actual joints from self.jointList (populated by jointStructure).
        Each segment (upper arm / forearm) is tagged independently.
        """
        super().finalize()
