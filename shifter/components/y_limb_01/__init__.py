# type: ignore
import ast
from typing import ClassVar

import mgear.pymaya as pm
from mgear.core import applyop, attribute, icon, node, primitive, string, transform, vector
from mgear.pymaya import datatypes
from mgear.shifter import component

from yrig.maya_api.node import QuatSlerpNode, QuatToEulerNode
from yrig.skin.split import tag_for_weight_split
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.transform import matrix_constraint
from yrig.transform.quat import create_swing_only_transform, twist_extract_quat

#############################################
# COMPONENT
#############################################


class Component(component.Main):
    """Shifter component Class"""

    # Override in subclasses to map guide locator names to canonical names
    GUIDE_MAP: ClassVar[dict[str, str]] = {}  # e.g. {"mid": "elbow", "end": "wrist"}
    WORLD_ALIGN_IK: bool = False

    # =====================================================
    # OBJECTS
    # =====================================================
    def addObjects(self) -> None:
        """Add all the objects needed to create the component."""

        self._add_common_setup()
        self._add_root_control()
        self._add_fk_controls()
        self._add_ik_upv()
        self._add_ik_controls()
        self._add_reference_objects()
        self._add_solver_chain()
        self._add_match_refs()
        self._add_swing_twist()
        self._add_mid_control()
        self._add_twist_chains()
        self._add_divisions()
        self._add_end_reference()
        self._add_bendy_controls()
        self._add_ik_visual_reference()

    def _add_common_setup(self) -> None:
        self.WIP = self.options["mode"]
        self.up_axis = pm.upAxis(q=True, axis=True)

        self.normal = self.getNormalFromPos(self.guide.apos)
        self.binormal = self.getBiNormalFromPos(self.guide.apos)

        self.length0 = vector.getDistance(self.guide.apos[0], self.guide.apos[1])
        self.length1 = vector.getDistance(self.guide.apos[1], self.guide.apos[2])
        self.length2 = vector.getDistance(self.guide.apos[2], self.guide.apos[3])

        # custom colors
        self.color_offset_fk = [1, 0.25, 0]  # orange

        self.jd_names = ast.literal_eval(self.settings["jointNamesDescription_custom"])

        self.root_guide = self.GUIDE_MAP.get("root", "root")
        self.mid_guide = self.GUIDE_MAP.get("mid", "mid")
        self.end_guide = self.GUIDE_MAP.get("end", "end")

    def _add_root_control(self) -> None:
        t = transform.getTransformFromPos(self.guide.apos[0])

        self.root_npo = primitive.addTransform(self.root, self.getName("root_npo"), t)
        self.root_ctl = self.addCtl(
            self.root_npo,
            "root_ctl",
            t,
            self.color_fk,
            "circle",
            w=self.length0 / 6,
            tp=self.parentCtlTag,
        )

    def _add_fk_controls(self) -> None:
        # FK Controlers -----------------------------------
        # FK 0
        t = transform.getTransformLookingAt(
            self.guide.apos[0],
            self.guide.apos[1],
            self.normal,
            "xz",
            self.negate,
        )

        t_npo = t

        self.fk0_npo = primitive.addTransform(self.root, self.getName("fk0_npo"), t_npo)

        self.fk0_ctl = self.addCtl(
            self.fk0_npo,
            "fk0_ctl",
            t,
            self.color_fk,
            "circle",
            w=self.size * 0.3,
            ro=datatypes.Vector([0, 0, 1.5708]),
            tp=self.parentCtlTag,
        )

        # FK 1
        t = transform.getTransformLookingAt(
            self.guide.apos[1],
            self.guide.apos[2],
            self.normal,
            "xz",
            self.negate,
        )

        t_npo = t

        self.fk1_npo = primitive.addTransform(self.fk0_ctl, self.getName("fk1_npo"), t_npo)

        self.fk1_ctl = self.addCtl(
            self.fk1_npo,
            "fk1_ctl",
            t,
            self.color_fk,
            "circle",
            w=self.size * 0.3,
            ro=datatypes.Vector([0, 0, 1.5708]),
            tp=self.fk0_ctl,
        )

        for f_ctl in [
            self.fk0_ctl,
            self.fk1_ctl,
        ]:
            attribute.setKeyableAttributes(f_ctl, ["tx", "ty", "tz", "ro", "rx", "ry", "rz"])

        # FK 2
        t = transform.getTransformLookingAt(
            self.guide.apos[2],
            self.guide.apos[3],
            self.normal,
            "xz",
            self.negate,
        )

        t_npo = t

        self.fk2_npo = primitive.addTransform(self.fk1_ctl, self.getName("fk2_npo"), t_npo)

        self.fk2_ctl = self.addCtl(
            self.fk2_npo,
            "fk2_ctl",
            t,
            self.color_fk,
            "circle",
            w=self.size * 0.3,
            ro=datatypes.Vector([0, 0, 1.5708]),
            tp=self.fk1_ctl,
        )

        attribute.setKeyableAttributes(self.fk2_ctl)

        self.fk_ctl = [self.fk0_ctl, self.fk1_ctl, self.fk2_ctl]
        for x in self.fk_ctl:
            attribute.setInvertMirror(x, ["tx", "ty", "tz"])

    def _add_ik_upv(self) -> None:
        # 1 bone chain for upv ref
        self.limbChainUpvRef = primitive.add2DChain(
            self.root,
            self.getName("limbUpvRef%s_jnt"),
            [self.guide.apos[0], self.guide.apos[2]],
            self.normal,
            False,
            self.WIP,
        )

        self.limbChainUpvRef[1].setAttr(
            "jointOrientZ", self.limbChainUpvRef[1].getAttr("jointOrientZ") * -1
        )

        # IK upv ---------------------------------

        # create tip point
        self.tip_ref = primitive.addTransform(
            self.limbChainUpvRef[0],
            self.getName("tip_ref"),
            self.limbChainUpvRef[0].getMatrix(worldSpace=True),
        )

        # create interpolate obj
        self.interpolate_lvl = primitive.addTransform(
            self.limbChainUpvRef[0],
            self.getName("int_lvl"),
            self.limbChainUpvRef[0].getMatrix(worldSpace=True),
        )

        # create roll npo and ctl
        self.roll_ctl_npo = primitive.addTransform(
            self.root,
            self.getName("roll_ctl_npo"),
            self.limbChainUpvRef[0].getMatrix(worldSpace=True),
        )
        if self.negate:
            off_x = -1.5708
        else:
            off_x = 1.5708
        off_y = 1.5708

        self.roll_ctl = self.addCtl(
            self.roll_ctl_npo,
            "roll_ctl",
            transform.getTransform(self.roll_ctl_npo),
            self.color_ik,
            "compas",
            w=self.size * 0.3,
            ro=datatypes.Vector([off_x, off_y, 0]),
            tp=self.parentCtlTag,
        )
        attribute.setKeyableAttributes(self.roll_ctl, ["rx"])
        # create upv control
        v = self.guide.apos[2] - self.guide.apos[0]
        v = self.normal ^ v
        v.normalize()
        v *= self.size * 0.8
        v += self.guide.apos[1]

        self.upv_cns = primitive.addTransformFromPos(self.root, self.getName("upv_cns"), v)

        self.upv_ctl = self.addCtl(
            self.upv_cns,
            "upv_ctl",
            transform.getTransform(self.upv_cns),
            self.color_ik,
            "diamond",
            w=self.size * 0.12,
            tp=self.parentCtlTag,
        )

        if self.settings["mirrorMid"]:
            if self.negate:
                self.upv_cns.rz.set(180)
                self.upv_cns.sy.set(-1)
        else:
            attribute.setInvertMirror(self.upv_ctl, ["tx"])
        attribute.setKeyableAttributes(self.upv_ctl, self.t_params)

    def _add_ik_controls(self) -> None:
        # IK Controlers -----------------------------------

        end_pos = self.guide.pos[self.end_guide]

        self.ik_cns = primitive.addTransformFromPos(self.root_ctl, self.getName("ik_cns"), end_pos)

        t = transform.getTransformFromPos(end_pos)
        self.ikcns_ctl = self.addCtl(
            self.ik_cns,
            "ikcns_ctl",
            t,
            self.color_ik,
            "null",
            w=self.size * 0.12,
            tp=self.parentCtlTag,
        )

        attribute.setInvertMirror(self.ikcns_ctl, ["tx", "ty", "tz"])
        # if self.settings["mirrorIK"] and self.negate:
        #     self.ik_cns.sx.set(-1)
        if self.WORLD_ALIGN_IK:
            m = transform.getTransformFromPos(self.guide.pos[self.end_guide])
        else:
            if self.negate:
                m = transform.getTransformLookingAt(
                    self.guide.pos[self.end_guide],
                    self.guide.pos["eff"],
                    self.normal,
                    "x-y",
                    True,
                )
                # if self.settings["mirrorIK"]:
                #     m = transform.setMatrixScale(m, [-1, 1, 1])
            else:
                m = transform.getTransformLookingAt(
                    self.guide.pos[self.end_guide],
                    self.guide.pos["eff"],
                    self.normal,
                    "xy",
                    False,
                )
        self.ik_ctl = self.addCtl(
            self.ikcns_ctl,
            "ik_ctl",
            m,
            self.color_ik,
            "cube",
            w=self.size * 0.15,
            h=self.size * 0.15,
            d=self.size * 0.15,
            tp=self.roll_ctl,
        )

        # if not self.settings["mirrorIK"]:
        #     attribute.setInvertMirror(self.ik_ctl, ["tx", "ry", "rz"])
        attribute.setKeyableAttributes(self.ik_ctl)
        # we use same as fk2_ctl
        ik_ref_t = transform.getTransformLookingAt(
            self.guide.apos[2],
            self.guide.apos[3],
            self.normal,
            "xz",
            self.negate,
        )
        self.ik_ctl_ref = primitive.addTransform(self.ik_ctl, self.getName("ikCtl_ref"), ik_ref_t)

        self.fk_ik_ctls = self.fk_ctl + [self.ik_ctl]

    def _add_reference_objects(self) -> None:
        # References --------------------------------------
        trnIK_ref = transform.getTransformLookingAt(
            self.guide.pos[self.end_guide],
            self.guide.pos["eff"],
            self.normal,
            "xz",
            self.negate,
        )
        self.ik_ref = primitive.addTransform(self.ik_ctl_ref, self.getName("ik_ref"), trnIK_ref)
        self.fk_ref = primitive.addTransform(self.fk_ctl[2], self.getName("fk_ref"), trnIK_ref)

    def _add_solver_chain(self) -> None:
        # Chain --------------------------------------------
        # The outputs of the ikfk2bone solver

        self.bone0 = primitive.addLocator(
            self.root_ctl,
            self.getName("0_bone"),
            transform.getTransform(self.fk_ctl[0]),
        )

        self.bone0_shp = self.bone0.getShape()
        self.bone0_shp.setAttr("localPositionX", self.n_factor * 0.5)
        self.bone0_shp.setAttr("localScale", 0.5, 0, 0)
        self.bone0.setAttr("sx", self.length0)
        bShape = self.bone0.getShape()
        bShape.setAttr("visibility", False)
        self.bone0_tr = primitive.addTransform(
            parent=self.root_ctl,
            name=self.getName("0_bone_tr"),
            m=transform.getTransform(self.fk_ctl[0]),
        )
        self.bone0_tr.setAttr("visibility", False)

        t = transform.getTransform(self.fk_ctl[1])
        self.bone1 = primitive.addLocator(self.root_ctl, self.getName("1_bone"), t)

        self.bone1_shp = self.bone1.getShape()
        self.bone1_shp.setAttr("localPositionX", self.n_factor * 0.5)
        self.bone1_shp.setAttr("localScale", 0.5, 0, 0)
        self.bone1.setAttr("sx", self.length1)
        bShape = self.bone1.getShape()
        bShape.setAttr("visibility", False)
        self.bone1_tr = primitive.addTransform(
            parent=self.root_ctl,
            name=self.getName("1_bone_tr"),
            m=transform.getTransform(self.fk0_ctl),
        )
        self.bone1_tr.setAttr("visibility", False)

        # Eff locator
        self.eff_loc = primitive.addTransform(
            self.root_ctl, self.getName("eff_loc"), transform.getTransform(self.fk2_ctl)
        )
        self.tws2_rot = self.eff_loc  # tws2_rot is needed for proper connection to the foot.

        # Mid bone1 ref — used as fallback when div1 == 0
        t = transform.getTransform(self.fk_ctl[1])
        self.mid_ref = primitive.addTransform(self.root, self.getName("mid_ref"), t)

    def _add_match_refs(self) -> None:
        # match IK FK references
        self.match_fk0_off = self.add_match_ref(self.fk_ctl[1], self.root, "matchFk0_npo", False)

        self.match_fk0 = self.add_match_ref(self.fk_ctl[0], self.match_fk0_off, "fk0_mth")

        self.match_fk1_off = self.add_match_ref(self.fk_ctl[2], self.root, "matchFk1_npo", False)

        self.match_fk1 = self.add_match_ref(self.fk_ctl[1], self.match_fk1_off, "fk1_mth")

        reference = self.ik_ctl

        self.match_fk2 = self.add_match_ref(self.fk_ctl[2], reference, "fk2_mth")

        self.match_ik = self.add_match_ref(self.ik_ctl, self.fk2_ctl, "ik_mth")

        self.match_ikUpv = self.add_match_ref(self.upv_ctl, self.fk0_ctl, "upv_mth")

    def _add_swing_twist(self) -> None:
        self.upper_swing = pm.PyNode(
            create_swing_only_transform(
                transform=str(self.bone0_tr),
                reference_space=str(self.root),
                axis="x",
                name=self.getName("upper_swing"),
            )
        )
        self.upper_twist = primitive.addTransform(
            parent=self.upper_swing,
            name=self.getName("upper_twist"),
            m=transform.getTransform(self.fk0_ctl),
        )

        self.lower_swing = pm.PyNode(
            create_swing_only_transform(
                transform=str(self.bone1_tr),
                reference_space=str(self.bone0_tr),
                axis="x",
                name=self.getName("lower_swing"),
                parent=self.root,
            )
        )
        lower_twist_pos = transform.getPositionFromMatrix(transform.getTransform(self.fk2_ctl))
        lower_twist_matrix = transform.setMatrixPosition(
            transform.getTransform(self.fk1_ctl), lower_twist_pos
        )
        self.end_eff_twist_out = primitive.addTransform(
            parent=self.eff_loc,
            name=self.getName("eff_twist_out"),
            m=lower_twist_matrix,
        )

        self.lower_twist = primitive.addTransform(
            parent=self.bone1_tr,
            name=self.getName("lower_twist"),
            m=lower_twist_matrix,
        )

    def _add_mid_control(self) -> None:
        # Mid control Locator

        tA = transform.getTransformLookingAt(
            self.guide.apos[0],
            self.guide.apos[1],
            self.normal,
            "xz",
            self.negate,
        )
        tA = transform.setMatrixPosition(tA, self.guide.apos[1])
        tB = transform.getTransformLookingAt(
            self.guide.apos[1],
            self.guide.apos[2],
            self.normal,
            "xz",
            self.negate,
        )

        t = transform.getInterpolateTransformMatrix(tA, tB)
        self.ctrn_loc = primitive.addTransform(self.root, self.getName("ctrn_loc"), t)

        # match IK FK references
        # Mid Controller ------------------------------------

        t = transform.getTransform(self.ctrn_loc)
        self.mid_cns = primitive.addTransform(self.ctrn_loc, self.getName("mid_cns"), t)
        self.mid_ctl = self.addCtl(
            self.mid_cns,
            "mid_ctl",
            t,
            self.color_ik,
            "sphere",
            w=self.size * 0.2,
            tp=self.parentCtlTag,
        )

        if self.settings["mirrorMid"]:
            if self.negate:
                self.mid_cns.rz.set(180)
                self.mid_cns.sz.set(-1)
        else:
            attribute.setInvertMirror(self.mid_ctl, ["tx", "ty", "tz"])
        attribute.setKeyableAttributes(self.mid_ctl, self.t_params)

    def _add_twist_chains(self) -> None:
        # Roll twist chain ---------------------------------
        # Upper
        self.upperChainPos = []
        ii = 1.0 / max(self.settings["div0"], 1)
        i = 0.0
        for _p in range(max(self.settings["div0"] + 1, 2)):
            p_vec = vector.linearlyInterpolate(
                self.guide.pos[self.root_guide], self.guide.pos[self.mid_guide], blend=i
            )
            self.upperChainPos.append(p_vec)
            i = i + ii

        self.upperTwistChain = primitive.add2DChain(
            self.root,
            self.getName("upperTwist%s_jnt"),
            self.upperChainPos,
            self.normal,
            self.negate,
            self.WIP,
        )

        # Lower
        self.lowerChainPos = []
        ii = 1.0 / max(self.settings["div1"], 1)
        i = 0.0
        for _p in range(max(self.settings["div1"] + 1, 2)):
            p_vec = vector.linearlyInterpolate(
                self.guide.pos[self.mid_guide], self.guide.pos[self.end_guide], blend=i
            )
            self.lowerChainPos.append(p_vec)
            i = i + ii

        self.lowerTwistChain = primitive.add2DChain(
            self.root,
            self.getName("lowerTwist%s_jnt"),
            self.lowerChainPos,
            self.normal,
            self.negate,
            self.WIP,
        )

    def _add_divisions(self) -> None:
        # Divisions ----------------------------------------
        # We have attribute least one division attribute the start, the end
        # and one for the mid control. + 2 for mid angle control
        self.extra_div = 2
        self.divisions = self.settings["div0"] + self.settings["div1"] + self.extra_div

        tagP = self.parentCtlTag
        self.tweak_ctl = []
        self.div_cns = []
        self.roll_offset = []

        # Track jnt_pos indices per segment for weight split tagging in finalize
        self.upper_jnt_indices = []
        self.lower_jnt_indices = []

        # joint Description Name
        jdn_upper = self.jd_names[0]
        jdn_lower = self.jd_names[1]
        jdn_upper_twist = self.jd_names[2]
        jdn_lower_twist = self.jd_names[3]

        for i in range(self.divisions):
            div_cns = primitive.addTransform(self.root_ctl, self.getName(f"div{i}_loc"))

            self.div_cns.append(div_cns)

            t = transform.getTransform(div_cns)
            tweak_ctl = self.addCtl(
                div_cns,
                f"tweak{i}_ctl",
                t,
                self.color_fk,
                "square",
                w=self.size * 0.15,
                d=self.size * 0.15,
                ro=datatypes.Vector([0, 0, 1.5708]),
                tp=tagP,
            )
            attribute.setKeyableAttributes(tweak_ctl)

            tagP = tweak_ctl
            self.tweak_ctl.append(tweak_ctl)

            roll_off = primitive.addTransform(tweak_ctl, self.getName(f"roll{i}_off"))

            self.roll_offset.append(roll_off)

            # setting the joints
            if i == 0:
                self.limb_root_base = roll_off

                self.upper_jnt_indices.append(len(self.jnt_pos))
                self.jnt_pos.append(
                    {
                        "obj": self.limb_root_base,
                        "name": jdn_upper,
                        "guide_relative": "root",
                        "data_contracts": "Ik",
                        "leaf_joint": self.settings["leafJoints"],
                    }
                )
                current_parent = "root"
                twist_name = jdn_upper_twist
                twist_idx = 1
                increment = 1

                # extra joint twist/swing
                if self.settings["div0"]:
                    self.jnt_pos.append(
                        {
                            "obj": roll_off,
                            "name": jdn_upper + "_swing",
                            "data_contracts": "Twist,Squash",
                            "newActiveJnt": current_parent,
                        }
                    )
            elif i == self.settings["div0"] + 1:
                self.lower_jnt_indices.append(len(self.jnt_pos))
                self.jnt_pos.append(
                    {
                        "obj": roll_off,
                        "name": jdn_lower,
                        "newActiveJnt": current_parent,
                        "guide_relative": self.mid_guide,
                        "data_contracts": "Ik",
                        "leaf_joint": self.settings["leafJoints"],
                    }
                )
                twist_name = jdn_lower_twist
                current_parent = self.mid_guide
                twist_idx = self.settings["div1"]
                increment = -1
            else:
                if twist_name == jdn_upper_twist:
                    self.upper_jnt_indices.append(len(self.jnt_pos))
                else:
                    self.lower_jnt_indices.append(len(self.jnt_pos))
                self.jnt_pos.append(
                    {
                        "obj": roll_off,
                        "name": string.replaceSharpWithPadding(twist_name, twist_idx),
                        "newActiveJnt": current_parent,
                        "data_contracts": "Twist,Squash",
                    }
                )
                twist_idx += increment
        self.divisions_end = current_parent

    def _add_end_reference(self) -> None:
        # End reference ------------------------------------
        # To help the deformation on the wrist/ankle

        # joint Description Name
        jdn_hand = self.jd_names[4]

        self.end_ref = primitive.addTransform(
            self.eff_loc,
            self.getName("end_ref"),
            transform.getTransform(self.eff_loc),
        )
        if self.negate:
            self.end_ref.attr("rz").set(180.0)

        self.eff_jnt_off = self.end_ref

        self.jnt_pos.append(
            {
                "obj": self.eff_jnt_off,
                "name": jdn_hand,
                "newActiveJnt": self.divisions_end,
                "guide_relative": "end",
                "data_contracts": "Ik",
                "leaf_joint": self.settings["leafJoints"],
            }
        )

    def _add_bendy_controls(self) -> None:
        # Bendy controls
        posA = transform.getTranslation(self.fk0_ctl)
        posB = transform.getTranslation(self.fk1_ctl)
        midpoint = vector.linearlyInterpolate(posA, posB, 0.5)
        t = transform.setMatrixPosition(transform.getTransform(self.fk0_ctl), midpoint)

        self.upperBendy_aim = primitive.addTransform(
            self.bone0,
            self.getName("upperBendy_aim"),
            self.fk_ctl[0].getMatrix(worldSpace=True),
        )
        self.upperBendy_pin = primitive.addTransform(
            self.upperBendy_aim,
            self.getName("upperBendy_pin"),
            t,
        )

        self.upperBendy_npo = primitive.addTransform(self.root, self.getName("upperBendy_npo"), t)
        self.upperBendy_ctl = self.addCtl(
            self.upperBendy_npo,
            "upperBendy_ctl",
            t,
            self.color_ik,
            "circle",
            w=self.size * 0.2,
            ro=datatypes.Vector(0, 0, 1.570796),
            tp=self.mid_ctl,
        )
        self.upperBendy_twist = primitive.addTransform(
            self.upperBendy_ctl,
            self.getName("upperBendy_twist"),
            t,
        )
        attribute.setKeyableAttributes(self.upperBendy_ctl)

        self.upperBendy_end = primitive.addTransform(
            self.bone0_tr,
            self.getName("upperBendy_end"),
            transform.setMatrixPosition(
                transform.getTransform(self.fk0_ctl), transform.getTranslation(self.fk1_ctl)
            ),
        )

        posA = transform.getTranslation(self.fk1_ctl)
        posB = transform.getTranslation(self.fk2_ctl)
        midpoint = vector.linearlyInterpolate(posA, posB, 0.5)
        t = transform.setMatrixPosition(transform.getTransform(self.fk1_ctl), midpoint)

        self.lowerBendy_aim = primitive.addTransform(
            self.bone1,
            self.getName("lowerBendy_aim"),
            transform.getTransformLookingAt(
                self.guide.apos[2],
                self.guide.apos[1],
                self.normal,
                "xz",
                self.negate,
            ),
        )
        self.lowerBendy_pin = primitive.addTransform(
            self.lowerBendy_aim, self.getName("lowerBendy_pin"), t
        )

        self.lowerBendy_npo = primitive.addTransform(self.root, self.getName("lowerBendy_npo"), t)
        self.lowerBendy_ctl = self.addCtl(
            self.lowerBendy_npo,
            "lowerBendy_ctl",
            t,
            self.color_ik,
            "circle",
            w=self.size * 0.2,
            ro=datatypes.Vector(0, 0, 1.570796),
            tp=self.mid_ctl,
        )
        self.lowerBendy_twist = primitive.addTransform(
            self.lowerBendy_ctl,
            self.getName("lowerBendy_twist"),
            t,
        )
        attribute.setKeyableAttributes(self.lowerBendy_ctl)

        t = self.mid_ctl.getMatrix(worldSpace=True)
        self.midBendy_npo = primitive.addTransform(self.mid_ctl, self.getName("midBendy_npo"), t)

        self.midBendy_ctl = self.addCtl(
            self.midBendy_npo,
            "midBendy_ctl",
            t,
            self.color_fk,
            "circle",
            w=self.size * 0.15,
            ro=datatypes.Vector(0, 0, 1.570796),
            tp=self.mid_ctl,
        )
        self.upper_mid_bendy_npo = primitive.addTransform(
            self.midBendy_ctl, self.getName("upperMidBendy_npo"), t
        )
        self.lower_mid_bendy_npo = primitive.addTransform(
            self.midBendy_ctl, self.getName("lowerMidBendy_npo"), t
        )
        if self.settings["mirrorMid"] and self.negate:
            self.midBendy_npo.rz.set(180)
            self.midBendy_npo.sz.set(-1)
        attribute.setKeyableAttributes(self.midBendy_ctl)

    def _add_ik_visual_reference(self) -> None:
        # add visual reference
        self.line_ref = icon.connection_display_curve(
            self.getName("visalRef"), [self.upv_ctl, self.mid_ctl]
        )

    # =====================================================
    # ATTRIBUTES
    # =====================================================
    def addAttributes(self) -> None:
        """Create the anim and setupr rig attributes for the component"""
        self._add_common_attributes()
        self._add_reference_array_attributes()
        self._add_proxy_attributes()

        self.resample_att = self.addSetupParam("resample", "Resample", "bool", True)
        self.absolute_att = self.addSetupParam("absolute", "Absolute", "bool", False)
        self.volume_blenshape_att = self.addSetupParam(
            "volume_blendshape", "Volume Blendshape", "double", 0, 0, 10
        )

    def _add_common_attributes(self) -> None:
        # Anim -------------------------------------------
        self.blend_att = self.addAnimParam(
            "blend", "Fk/Ik Blend", "double", self.settings["blend"], 0, 1
        )
        self.roll_att = self.addAnimParam("roll", "IK Roll", "double", 0, -180, 180)
        self.scale_att = self.addAnimParam("ikscale", "IK Scale", "double", 1, 0.001, 10)

        self.maxstretch_att = self.addAnimParam(
            "maxstretch",
            "Max Stretch",
            "double",
            self.settings["maxstretch"],
            1,
            100,
        )

        self.slide_att = self.addAnimParam("slide", "Slide", "double", 0.5, 0, 1)

        self.softness_att = self.addAnimParam("softness", "Softness", "double", 0, 0, 1)

        self.reverse_att = self.addAnimParam("reverse", "Reverse", "double", 0, 0, 1)

        self.roundness_att = self.addAnimParam("roundness", "Roundness", "double", 0, 0, 1)

        self.volume_att = self.addAnimParam("volume", "Volume Joint Scale", "double", 0, 0, 1)

        self.bendyVis_att = self.addAnimParam("Bendy_vis", "Bendy vis", "bool", False)

        self.midBendyVis_att = self.addAnimParam("midBendy_vis", "Mid Bendy vis", "bool", False)

        self.upvAimVis_att = self.addAnimParam("UpvAim_vis", "IK Pole Vector vis", "bool", True)
        self.upvCtlVis_att = self.addAnimParam("UpvCtl_vis", "IK Roll Ctl vis", "bool", False)
        self.tweakVis_att = self.addAnimParam("Tweak_vis", "Tweak Vis", "bool", False)

        self.midCtl_att = self.addAnimParam("mid_ctl_vis", "Mid Ctl Vis", "bool", False)
        self.ikCnsCtl_att = self.addAnimParam("ik_cns_ctl_vis", "IK Cns Ctl Vis", "bool", False)

    def _add_reference_array_attributes(self) -> None:
        # Ref
        if self.settings["ikrefarray"]:
            ref_names = self.settings["ikrefarray"].split(",")
            if len(ref_names) > 1:
                self.ikref_att = self.addAnimEnumParam(
                    "ikref",
                    "Ik Ref",
                    0,
                    self.settings["ikrefarray"].split(","),
                )

        if self.settings["upvrefarray"]:
            ref_names = self.settings["upvrefarray"].split(",")
            ref_names = ["Auto"] + ref_names
            if len(ref_names) > 1:
                self.upvref_att = self.addAnimEnumParam("upvref", "UpV Ref", 0, ref_names)

        if self.settings["pinrefarray"]:
            ref_names = self.settings["pinrefarray"].split(",")
            ref_names = ["Auto"] + ref_names
            if len(ref_names) > 1:
                self.pin_att = self.addAnimEnumParam("midref", "Mid Ref", 0, ref_names)

    def _add_proxy_attributes(self) -> None:
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
        self._setup_ik_upv()
        self._setup_control_vis()
        self._setup_control_rotation_orders()
        self._setup_ik_solver()
        self._setup_swing_twist()
        self._setup_bendy_controls()
        self._setup_roll_control()
        self._setup_twist_chains()
        self._setup_ik_fk_match()
        self._setup_joints()

    def _setup_ik_upv(self) -> None:
        # 1 bone chain Upv ref ======================================
        self.ikHandleUpvRef = primitive.addIkHandle(
            self.root,
            self.getName("ikHandleLimbChainUpvRef"),
            self.limbChainUpvRef,
            "ikSCsolver",
        )
        pm.pointConstraint(self.ik_ctl, self.ikHandleUpvRef)
        # pm.parentConstraint(self.limbChainUpvRef[0], self.upv_cns, mo=True)
        # handle special case for full mirror behaviour negating
        # scaleY axis to -1
        if self.upv_cns.sy.get() < 0:
            references = []
            for x in [self.limbChainUpvRef[0]]:
                ref_trans_name = self.upv_cns.getName() + "_" + x.getName() + "_space_ref"
                ref_trans = primitive.addTransform(
                    x,
                    ref_trans_name,
                )
                transform.matchWorldTransform(self.upv_cns, ref_trans)
                references.append(ref_trans)
            self.ikH_parCns = pm.parentConstraint(references[0], self.upv_cns, mo=True)
            self.ikH_cns_driver = references[0]
        else:
            self.ikH_parCns = pm.parentConstraint(self.limbChainUpvRef[0], self.upv_cns, mo=True)
            self.ikH_cns_driver = self.limbChainUpvRef[0]

    def _setup_control_vis(self) -> None:
        # Visibilities -------------------------------------
        # fk
        fkvis_node = node.createReverseNode(self.blend_att)

        for shp in self.fk0_ctl.getShapes():
            pm.connectAttr(fkvis_node + ".outputX", shp.attr("visibility"))
        for shp in self.fk1_ctl.getShapes():
            pm.connectAttr(fkvis_node + ".outputX", shp.attr("visibility"))
        for shp in self.fk2_ctl.getShapes():
            pm.connectAttr(fkvis_node + ".outputX", shp.attr("visibility"))

        # ik
        for shp in self.upv_ctl.getShapes():
            pm.connectAttr(self.blend_att, shp.attr("visibility"))

        add_node = node.createPlusMinusAverage1D([self.blend_att, self.ikCnsCtl_att])
        cond_node = node.createConditionNode(add_node.output1D, 1.5, 3, 1, 0)
        cond_node.colorIfFalseR.set(0)
        for shp in self.ikcns_ctl.getShapes():
            pm.connectAttr(cond_node.outColorR, shp.attr("visibility"))

        for shp in self.ik_ctl.getShapes():
            pm.connectAttr(self.blend_att, shp.attr("visibility"))
        for shp in self.line_ref.getShapes():
            pm.connectAttr(self.blend_att, shp.attr("visibility"))

        for shp in self.roll_ctl.getShapes():
            pm.connectAttr(self.blend_att, shp.attr("visibility"))

        pm.connectAttr(self.upvAimVis_att, self.upv_cns.visibility)
        pm.connectAttr(self.upvCtlVis_att, self.roll_ctl_npo.visibility)

        for tweak_ctl in self.tweak_ctl:
            for shp in tweak_ctl.getShapes():
                pm.connectAttr(self.tweakVis_att, shp.attr("visibility"))

        for shp in self.mid_ctl.getShapes():
            pm.connectAttr(self.midCtl_att, shp.attr("visibility"))

        # Bendy controls vis
        for shp in self.upperBendy_ctl.getShapes():
            pm.connectAttr(self.bendyVis_att, shp.attr("visibility"))
        for shp in self.lowerBendy_ctl.getShapes():
            pm.connectAttr(self.bendyVis_att, shp.attr("visibility"))
        for shp in self.midBendy_ctl.getShapes():
            pm.connectAttr(self.midBendyVis_att, shp.attr("visibility"))

    def _setup_control_rotation_orders(self) -> None:
        attribute.setRotOrder(self.fk0_ctl, "XZY")
        attribute.setRotOrder(self.fk1_ctl, "XYZ")
        attribute.setRotOrder(self.fk2_ctl, "YZX")
        attribute.setRotOrder(self.ik_ctl, "XYZ")

    def _setup_ik_solver(self) -> None:
        # IK Solver -----------------------------------------
        out = [self.bone0, self.bone1, self.ctrn_loc, self.eff_loc]
        o_node = applyop.gear_ikfk2bone_op(
            out,
            self.root_ctl,
            self.ik_ref,
            self.upv_ctl,
            self.fk_ctl[0],
            self.fk_ctl[1],
            self.fk_ref,
            self.length0,
            self.length1,
            self.negate,
        )
        # NOTE: Ideally we should not change hierarchy or move object after
        # object generation method. But is much easier this way since every
        # part is in the final and correct position
        # after the  ctrn_loc is in the correct position with the ikfk2bone op

        matrix_constraint(
            str(self.bone0), str(self.bone0_tr), keep_offset=False, scale=False, shear=False
        )
        matrix_constraint(
            str(self.bone1), str(self.bone1_tr), keep_offset=False, scale=False, shear=False
        )

        # connect mid ref
        cns = pm.parentConstraint(self.bone1, self.mid_ref, mo=False)
        if self.negate and self.settings["div1"]:
            pm.setAttr(cns + ".target[0].targetOffsetRotateZ", 180)

        # scale: this fix the scalin popping issue
        intM_node = applyop.gear_intmatrix_op(
            self.fk2_ctl.attr("worldMatrix"),
            self.ik_ctl_ref.attr("worldMatrix"),
            o_node.attr("blend"),
        )

        mulM_node = applyop.gear_mulmatrix_op(
            intM_node.attr("output"), self.eff_loc.attr("parentInverseMatrix")
        )

        dm_node = node.createDecomposeMatrixNode(mulM_node.attr("output"))
        dm_node.attr("outputScale") >> self.eff_loc.attr("scale")

        pm.connectAttr(self.blend_att, o_node + ".blend")
        if self.negate:
            mulVal = -1
            rollMulVal = 1
        else:
            mulVal = 1
            rollMulVal = -1
        roll_m_node = node.createMulNode(self.roll_att, mulVal)
        roll_m_node2 = node.createMulNode(self.roll_ctl.attr("rx"), rollMulVal)
        node.createPlusMinusAverage1D(
            [roll_m_node.outputX, roll_m_node2.outputX],
            operation=1,
            output=o_node + ".roll",
        )
        pm.connectAttr(self.scale_att, o_node + ".scaleA")
        pm.connectAttr(self.scale_att, o_node + ".scaleB")
        pm.connectAttr(self.maxstretch_att, o_node + ".maxstretch")
        pm.connectAttr(self.slide_att, o_node + ".slide")
        pm.connectAttr(self.softness_att, o_node + ".softness")
        pm.connectAttr(self.reverse_att, o_node + ".reverse")

        # point constrain tip reference
        pm.pointConstraint(self.ik_ctl, self.tip_ref, mo=False)

    def _setup_swing_twist(self) -> None:
        matrix_constraint(
            str(self.midBendy_ctl),
            str(self.upper_twist),
            keep_offset=False,
            rotate=False,
            scale=False,
            shear=False,
        )
        matrix_constraint(
            str(self.eff_loc),
            str(self.lower_twist),
            keep_offset=False,
            rotate=False,
            scale=False,
            shear=False,
        )

        upper_twist_quat = twist_extract_quat(str(self.bone1), str(self.upper_swing), axis="x")
        upper_twist_euler = QuatToEulerNode.create(f"{self.bone0}_twist")
        upper_twist_euler.input_quat.connect_from(upper_twist_quat)
        upper_twist_euler.output_rotate.x.connect_to(f"{self.upper_twist}.rotateX")
        upper_twist_mid = QuatSlerpNode.create(f"{upper_twist_quat}_blend")
        upper_twist_mid.input2_quat.connect_from(upper_twist_quat)
        upper_twist_mid.input_t.set(0.5)
        upper_twist_mid_euler = QuatToEulerNode.create(f"{upper_twist_mid}_euler")
        upper_twist_mid.output_quat.connect_to(upper_twist_mid_euler.input_quat)
        upper_twist_mid_euler.output_rotate.x.connect_to(f"{self.upperBendy_twist}.rotateX")

        lower_twist_quat = twist_extract_quat(
            str(self.end_eff_twist_out), str(self.bone1_tr), axis="x"
        )
        lower_twist_euler = QuatToEulerNode.create(f"{self.bone1}_twist")
        lower_twist_euler.input_quat.connect_from(lower_twist_quat)
        lower_twist_euler.output_rotate.x.connect_to(f"{self.lower_twist}.rotateX")
        lower_twist_mid = QuatSlerpNode.create(f"{lower_twist_quat}_blend")
        lower_twist_mid.input2_quat.connect_from(lower_twist_quat)
        lower_twist_mid.input_t.set(0.5)
        lower_twist_mid_euler = QuatToEulerNode.create(f"{lower_twist_mid}_euler")
        lower_twist_mid.output_quat.connect_to(lower_twist_mid_euler.input_quat)
        lower_twist_mid_euler.output_rotate.x.connect_to(f"{self.lowerBendy_twist}.rotateX")

    def _setup_bendy_controls(self) -> None:
        matrix_constraint(
            str(self.upper_twist),
            str(self.upper_mid_bendy_npo),
            translate=False,
            rotate=True,
            scale=False,
            shear=False,
        )
        matrix_constraint(
            str(self.bone1_tr),
            str(self.lower_mid_bendy_npo),
            translate=False,
            rotate=True,
            scale=False,
            shear=False,
        )

        matrix_spline_from_transforms(
            name=f"{self.upperBendy_ctl}_aim_spline",
            cv_transforms=[str(self.bone0_tr), str(self.midBendy_npo)],
            pinned_transforms=[str(self.upperBendy_npo)],
            twist=False,
            parent=str(self.upper_swing),
            primary_axis=(1, 0, 0) if not self.negate else (-1, 0, 0),
            arc_length=False,
            degree=1,
            stretch=False,
        )
        matrix_spline_from_transforms(
            name=f"{self.lowerBendy_ctl}_aim_spline",
            cv_transforms=[str(self.midBendy_npo), str(self.lower_twist)],
            pinned_transforms=[str(self.lowerBendy_npo)],
            twist=False,
            parent=str(self.bone1_tr),
            primary_axis=(1, 0, 0) if not self.negate else (-1, 0, 0),
            arc_length=False,
            degree=1,
            stretch=False,
        )

    def _setup_roll_control(self) -> None:
        # interpolate transform  mid point locator
        int_matrix = applyop.gear_intmatrix_op(
            self.limbChainUpvRef[0].attr("worldMatrix"),
            self.tip_ref.attr("worldMatrix"),
            0.5,
        )
        applyop.gear_mulmatrix_op(
            int_matrix.attr("output"),
            self.interpolate_lvl.attr("parentInverseMatrix[0]"),
            self.interpolate_lvl,
        )

        # match roll ctl npo to ctrn_loc current transform (so correct orient)
        transform.matchWorldTransform(self.ctrn_loc, self.roll_ctl_npo)

        # match roll ctl npo to interpolate transform current position
        pos = self.interpolate_lvl.getTranslation(space="world")
        self.roll_ctl_npo.setTranslation(pos, space="world")

        # parent constraint roll control npo to interpolate trans
        pm.parentConstraint(self.interpolate_lvl, self.roll_ctl_npo, mo=True)

    def _setup_twist_chains(self) -> None:
        # spline IK for  twist jnts
        cns_list = [
            self.upper_swing,
            self.upperBendy_twist,
            self.upper_mid_bendy_npo,
        ]

        self.upper_twist_spline = matrix_spline_from_transforms(
            name=f"{self.bone0_tr}_twist",
            parent=str(self.bone0_tr),
            cv_transforms=[str(transform) for transform in cns_list],
            primary_axis=(1, 0, 0) if not self.negate else (-1, 0, 0),
            secondary_axis=(0, 0, 1) if not self.negate else (0, 0, -1),
            degree=2,
            pinned_transforms=[str(transform) for transform in self.upperTwistChain],
            padded=False,
        )

        cns_list = [
            self.lower_mid_bendy_npo,
            self.lowerBendy_twist,
            self.lower_twist,
        ]
        self.lower_twist_spline = matrix_spline_from_transforms(
            name=f"{self.bone1_tr}_twist",
            parent=str(self.bone1_tr),
            cv_transforms=[str(transform) for transform in cns_list],
            primary_axis=(1, 0, 0) if not self.negate else (-1, 0, 0),
            secondary_axis=(0, 0, 1) if not self.negate else (0, 0, -1),
            degree=2,
            pinned_transforms=[str(transform) for transform in self.lowerTwistChain],
            padded=False,
        )

    def _setup_ik_fk_match(self) -> None:
        # TODO: check for a more clean and elegant solution instead of re-match
        # the world matrix again
        transform.matchWorldTransform(self.fk_ctl[0], self.match_fk0_off)
        transform.matchWorldTransform(self.fk_ctl[1], self.match_fk1_off)
        transform.matchWorldTransform(self.fk_ctl[0], self.match_fk0)
        transform.matchWorldTransform(self.fk_ctl[1], self.match_fk1)

        # match IK/FK ref
        pm.parentConstraint(self.bone0, self.match_fk0_off, mo=True)
        pm.parentConstraint(self.bone1, self.match_fk1_off, mo=True)

    def _setup_divisions(self) -> None:
        # Divisions ----------------------------------------
        # attribute 0 or 1 the division will follow exactly the rotation of
        # the controler.. and we wont have this nice bendy + roll
        div_offset = int(self.extra_div / 2)
        for i, div_cns in enumerate(self.div_cns):
            if i == 0 or i < (self.settings["div0"] + div_offset):
                mulmat_node = applyop.gear_mulmatrix_op(
                    self.upperTwistChain[i] + ".worldMatrix",
                    div_cns + ".parentInverseMatrix",
                )
            elif i == (self.settings["div0"] + div_offset) and self.settings["div1"] == 0:
                mulmat_node = applyop.gear_mulmatrix_op(
                    self.mid_ref + ".worldMatrix",
                    div_cns + ".parentInverseMatrix",
                )
            else:
                ftc = self.lowerTwistChain[i - (self.settings["div0"] + div_offset)]
                mulmat_node = applyop.gear_mulmatrix_op(
                    ftc + ".worldMatrix", div_cns + ".parentInverseMatrix"
                )

            dm_node = node.createDecomposeMatrixNode(mulmat_node + ".output")
            pm.connectAttr(dm_node + ".outputTranslate", div_cns + ".translate")
            pm.connectAttr(dm_node + ".outputRotate", div_cns + ".rotate")
            pm.connectAttr(dm_node + ".outputScale", div_cns + ".scale")
            pm.connectAttr(dm_node + ".outputShear", div_cns + ".shear")

    def _setup_joints(self) -> None:
        self._setup_divisions()
        # force translation for mid joint to mid ctl
        lastLimbDiv = None
        if not self.settings["div0"]:
            lastLimbDiv = self.div_cns[1]
        elif not self.settings["div1"]:
            lastLimbDiv = self.div_cns[-1]

        if lastLimbDiv:
            applyop.gear_mulmatrix_op(
                self.midBendy_ctl.worldMatrix,
                lastLimbDiv.parentInverseMatrix,
                lastLimbDiv,
                "t",
            )

    # =====================================================
    # CONNECTOR
    # =====================================================

    def setRelation(self) -> None:
        """Set the relation beetween object from guide to rig"""
        offset = int(self.extra_div / 2)
        self.relatives[self.root_guide] = self.div_cns[0]
        self.relatives[self.mid_guide] = self.div_cns[self.settings["div0"] + offset]
        self.relatives[self.end_guide] = self.div_cns[-1]
        self.relatives["eff"] = self.eff_loc

        self.jointRelatives["root"] = 0
        # we need to account with the extra joint for swing
        if self.settings["div0"]:
            offset = offset + 1

        self.jointRelatives[self.mid_guide] = self.settings["div0"] + offset
        self.jointRelatives[self.end_guide] = len(self.div_cns) - offset
        self.jointRelatives["eff"] = -1

        self.controlRelatives[self.root_guide] = self.fk0_ctl
        self.controlRelatives[self.mid_guide] = self.fk1_ctl
        self.controlRelatives[self.end_guide] = self.fk2_ctl
        self.controlRelatives["eff"] = self.fk2_ctl

    def addConnection(self) -> None:
        """Add more connection definition to the set"""

    def connect_standard(self) -> None:
        """standard connection definition for the component"""
        self.parent.addChild(self.root)
        self._connect_reference_array()

    def _connect_reference_array(self) -> None:
        # Set the Ik Reference
        self.connectRef(self.settings["ikrefarray"], self.ik_cns)
        self.connectRef(self.settings["upvrefarray"], self.upv_cns, True)

        if self.settings["pinrefarray"]:
            self.connectRef2(
                self.settings["pinrefarray"],
                self.mid_cns,
                self.pin_att,
                [self.ctrn_loc],
                False,
                ["Auto"],
            )

    def finalize(self) -> None:
        """Tag split joints for automatic weight splitting.

        Uses the jnt_pos indices recorded during addObjects to look up
        the actual joints from self.jointList (populated by jointStructure).
        Each segment (upper / lower) is tagged independently.
        """
        if self.settings["weight_split_tag"]:
            for segment_indices in (self.upper_jnt_indices, self.lower_jnt_indices):
                if len(segment_indices) > 1:
                    segment_joints = [self.jointList[index].name() for index in segment_indices]
                    tag_for_weight_split(
                        influence=segment_joints[0],
                        split_influences=segment_joints,
                        degree=self.settings["weight_split_degree"],
                    )

        super().finalize()
