# type: ignore
import ast
from itertools import chain

import mgear.pymaya as pm
from mgear.core import attribute, primitive, string, transform
from mgear.pymaya import datatypes
from mgear.shifter import component

from yrig.maya_api.node import CurveInfoNode, MultiplyNode, SubtractNode
from yrig.spline import pin_to_matrix_spline
from yrig.spline.curve import bound_curve_from_transforms
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms

#############################################
# COMPONENT
#############################################


class Component(component.Main):
    """Shifter component Class"""

    # =====================================================
    # OBJECTS
    # =====================================================
    def addObjects(self):
        """Add all the objects needed to create the component."""

        self.WIP = self.options["mode"]
        self.normal = self.guide.blades["blade"].z * -1
        self.up_axis = pm.upAxis(q=True, axis=True)

        # joint Description Names
        jd_names = ast.literal_eval(self.settings["jointNamesDescription_custom"])
        jdn_spine = jd_names[1]

        # Rotation-only chain (for spine length adjustment)
        self.rotation_group = primitive.addTransform(self.root, self.getName("rotation_grp"))
        self.rotation_group.setAttr("visibility", False)
        self.transform2Lock.append(self.rotation_group)
        # Main Controllers ------------------------------------
        t = transform.getTransformLookingAt(
            self.guide.pos["spineBase"],
            self.guide.pos["chest"],
            self.guide.blades["blade"].z * -1,
            "yx",
            self.negate,
        )
        if self.settings["IKWorldOri"]:
            print("World")
            ik_t = datatypes.TransformationMatrix()
            ik_t = transform.setMatrixPosition(ik_t, self.guide.pos["spineBase"])
        else:
            ik_t = t

        # Hip Control and Tangents
        hip_name = "hip"
        hip_transform = transform.setMatrixPosition(ik_t, self.guide.pos["hipPivot"])
        hip_control_offset = hip_transform.inverse() * (
            self.guide.pos["spineBase"] - self.guide.pos["hipPivot"]
        )
        self.hip_npo = primitive.addTransform(
            self.root, self.getName(f"{hip_name}_npo"), hip_transform
        )
        self.hip_ctl = self.addCtl(
            self.hip_npo,
            f"{hip_name}_ctl",
            hip_transform,
            self.color_ik,
            "circle",
            w=self.size * 0.8,
            tp=self.parentCtlTag,
            po=hip_control_offset,
        )
        attribute.setRotOrder(self.hip_ctl, "YZX")
        attribute.setInvertMirror(self.hip_ctl, ["tx", "ry", "rz"])
        spine_base_transform = transform.setMatrixPosition(ik_t, self.guide.pos["spineBase"])
        self.spine_base = primitive.addTransform(
            self.hip_ctl, self.getName("spine_base"), spine_base_transform
        )
        self.transform2Lock.append(self.spine_base)
        hip_tan_transform = transform.setMatrixPosition(ik_t, self.guide.pos["tan0"])
        self.mid_spline_hip_tan = primitive.addTransform(
            self.hip_ctl, self.getName("mid_spline_hip_tan"), hip_tan_transform
        )
        self.transform2Lock.append(self.mid_spline_hip_tan)
        self.spine_base_rotation = primitive.addTransform(
            self.rotation_group, self.getName("spine_base_rotation"), spine_base_transform
        )
        self.transform2Lock.append(self.spine_base_rotation)
        self.hip_tan_rotation = primitive.addTransform(
            self.rotation_group, self.getName("hip_tan_rotation"), hip_tan_transform
        )
        self.transform2Lock.append(self.hip_tan_rotation)

        # Mid control and tangents
        mid_name = "mid"
        mid_transform = transform.setMatrixPosition(ik_t, self.guide.pos["chestPivot"])
        self.mid_npo = primitive.addTransform(
            self.root, self.getName(f"{mid_name}_npo"), mid_transform
        )
        self.mid_twist = primitive.addTransform(
            self.mid_npo, self.getName(f"{mid_name}_twist"), mid_transform
        )
        self.mid_ctl = self.addCtl(
            self.mid_twist,
            f"{mid_name}_ctl",
            mid_transform,
            self.color_ik,
            "circle",
            w=self.size * 0.8,
            tp=self.parentCtlTag,
        )
        attribute.setRotOrder(self.mid_ctl, "YZX")
        attribute.setInvertMirror(self.mid_ctl, ["tx", "ry", "rz"])
        self.transform2Lock.append(self.mid_npo)
        self.hip_tan = primitive.addTransform(
            self.mid_ctl, self.getName("hip_tan"), hip_tan_transform
        )
        self.transform2Lock.append(self.hip_tan)

        # Torso Control
        torso_name = "torso"
        torso_transform = transform.setMatrixPosition(ik_t, self.guide.pos["spineBase"])
        self.torso_rotation = primitive.addTransform(
            self.rotation_group, self.getName(f"{torso_name}_rotation"), torso_transform
        )
        self.transform2Lock.append(self.torso_rotation)
        self.torso_npo = primitive.addTransform(
            self.root, self.getName(f"{torso_name}_npo"), torso_transform
        )
        self.torso_ctl = self.addCtl(
            self.torso_npo,
            f"{torso_name}_ctl",
            torso_transform,
            self.color_fk,
            "circle",
            w=(self.size),
            tp=self.parentCtlTag,
        )
        attribute.setRotOrder(self.torso_ctl, "YZX")
        attribute.setInvertMirror(self.torso_ctl, ["tx", "ry", "rz"])

        # Chest Control
        chest_name = "chest"
        chest_transform = transform.setMatrixPosition(ik_t, self.guide.pos["chestPivot"])
        chest_control_offset = chest_transform.inverse() * (
            self.guide.pos["chest"] - self.guide.pos["chestPivot"]
        )
        self.chest_rotation = primitive.addTransform(
            self.torso_rotation, self.getName(f"{chest_name}_rotation"), chest_transform
        )
        self.transform2Lock.append(self.chest_rotation)
        self.chest_npo = primitive.addTransform(
            self.torso_ctl, self.getName(f"{chest_name}_npo"), chest_transform
        )
        self.chest_ctl = self.addCtl(
            self.chest_npo,
            f"{chest_name}_ctl",
            chest_transform,
            self.color_ik,
            "compas",
            w=self.size * 0.8,
            tp=self.parentCtlTag,
            po=chest_control_offset,
        )
        attribute.setRotOrder(self.chest_ctl, "YZX")
        attribute.setInvertMirror(self.chest_ctl, ["tx", "ry", "rz"])

        # Chest IK Control
        chest_ik_name = "chest_ik"
        chest_ik_transform = transform.setMatrixPosition(ik_t, self.guide.pos["chestPivot"])
        chest_ik_control_offset = chest_ik_transform.inverse() * (
            self.guide.pos["spineTop"] - self.guide.pos["chestPivot"]
        )
        self.chest_ik_rotation = primitive.addTransform(
            self.chest_rotation, self.getName(f"{chest_ik_name}_rotation"), chest_ik_transform
        )
        self.transform2Lock.append(self.chest_ik_rotation)
        self.chest_ik_npo = primitive.addTransform(
            self.chest_ctl, self.getName(f"{chest_ik_name}_npo"), chest_ik_transform
        )
        self.chest_ik_ctl = self.addCtl(
            self.chest_ik_npo,
            f"{chest_ik_name}_ctl",
            chest_ik_transform,
            self.color_ik,
            "circle",
            w=self.size * 0.8,
            tp=self.chest_ctl,
            po=chest_ik_control_offset - datatypes.Vector(0, self.size * 0.1, 0),
        )
        attribute.setRotOrder(self.chest_ik_ctl, "YZX")
        attribute.setInvertMirror(self.chest_ik_ctl, ["tx", "ry", "rz"])

        chest_tan_transform = transform.setMatrixPosition(ik_t, self.guide.pos["tan1"])
        self.mid_spline_chest_tan = primitive.addTransform(
            self.chest_ik_ctl, self.getName("mid_spline_chest_tan"), chest_tan_transform
        )
        self.transform2Lock.append(self.mid_spline_chest_tan)
        self.chest_tan = primitive.addTransform(
            self.mid_ctl, self.getName("chest_tan"), chest_tan_transform
        )
        self.transform2Lock.append(self.chest_tan)

        # Chest Top Control and Tangents
        chest_top_name = "chest_top"
        chest_top_transform = transform.setMatrixPosition(ik_t, self.guide.pos["spineTop"])
        self.chest_top_npo = primitive.addTransform(
            self.chest_ik_ctl, self.getName(f"{chest_ik_name}_npo"), chest_top_transform
        )
        self.chest_top_length_adjust = primitive.addTransform(
            self.chest_top_npo, self.getName(f"{chest_ik_name}_length_adjust"), chest_top_transform
        )
        self.chest_top_ctl = self.addCtl(
            self.chest_top_length_adjust,
            f"{chest_top_name}_ctl",
            chest_top_transform,
            self.color_ik,
            "circle",
            w=self.size * 0.8,
            tp=self.chest_ik_ctl,
        )
        attribute.setRotOrder(self.chest_top_ctl, "YZX")
        attribute.setInvertMirror(self.chest_top_ctl, ["tx", "ry", "rz"])
        self.transform2Lock.append(self.chest_top_npo)
        spine_top_transform = transform.setMatrixPosition(ik_t, self.guide.pos["spineTop"])
        self.spine_top = primitive.addTransform(
            self.chest_top_ctl, self.getName("spine_top"), spine_top_transform
        )
        self.transform2Lock.append(self.spine_top)

        self.chest_tan_rotation = primitive.addTransform(
            self.chest_ik_rotation, self.getName("chest_tan_rotation"), chest_tan_transform
        )
        self.transform2Lock.append(self.chest_tan_rotation)
        self.spine_top_rotation = primitive.addTransform(
            self.chest_ik_rotation, self.getName("spine_top_rotation"), spine_top_transform
        )
        self.transform2Lock.append(self.spine_top_rotation)

        # Pin Mid Control so it's in the right spot and create the length curve
        self.mid_matrix_spline = matrix_spline_from_transforms(
            name=self.getName("mid_spline"),
            cv_transforms=[
                self.spine_base,
                self.mid_spline_hip_tan,
                self.mid_spline_chest_tan,
                self.spine_top,
            ],
            parent=self.root,
        )
        pin_to_matrix_spline(
            self.mid_matrix_spline,
            self.mid_npo,
            stretch=False,
            parameter=0.5,
            normalize_parameter=True,
            primary_axis=(0, 1, 0),
            secondary_axis=(1, 0, 0),
        )
        self.length_curve = bound_curve_from_transforms(
            transforms=[
                self.spine_base_rotation,
                self.hip_tan_rotation,
                self.chest_tan_rotation,
                self.spine_top_rotation,
            ],
            name=self.getName("length_ref"),
            parent=self.root,
        )

        # Division -----------------------------------------
        self.bind_spline = matrix_spline_from_transforms(
            name=self.getName("bind_spline"),
            cv_transforms=[self.spine_base, self.hip_tan, self.chest_tan, self.spine_top],
            parent=self.root,
            pinned_transforms=int(self.settings["division"]),
            padded=False,
            primary_axis=(0, 1, 0),
            secondary_axis=(1, 0, 0),
        )
        print(self.bind_spline.pinned_transforms)
        self.base_connection = pm.PyNode(self.bind_spline.pinned_transforms[0])
        self.top_connection = pm.PyNode(self.bind_spline.pinned_transforms[-1])
        start_transform = self.spine_base
        mid_pin_transforms = (
            pm.PyNode(transform) for transform in self.bind_spline.pinned_transforms[1:-1]
        )
        end_transform = self.spine_top
        for i, pin in enumerate(chain((start_transform,), mid_pin_transforms, (end_transform,))):
            self.jnt_pos.append(
                {
                    "obj": pin,
                    "name": string.replaceSharpWithPadding(jdn_spine, i + 1),
                    "data_contracts": "Twist",
                    "leaf_joint": self.settings["leafJoints"],
                }
            )

    def addAttributes(self):
        # Anim -------------------------------------------
        self.preserve_length_att = self.addAnimParam(
            "preserveLength", "Preserve Length", "double", self.settings["preserve_length"], 0, 1
        )

    # =====================================================
    # OPERATORS
    # =====================================================
    def addOperators(self):
        """Create operators and set the relations for the component rig

        Apply operators, constraints, expressions to the hierarchy.
        In order to keep the code clean and easier to debug,
        we shouldn't create any new object in this method.

        """
        control_rotation_map = {
            self.torso_ctl: self.torso_rotation,
            self.chest_ctl: self.chest_rotation,
            self.chest_ik_ctl: self.chest_ik_rotation,
        }
        for control, rotation_transform in control_rotation_map.items():
            pm.connectAttr(f"{control}.rotate", f"{rotation_transform}.rotate")
            pm.connectAttr(f"{control}.rotateOrder", f"{rotation_transform}.rotateOrder")

        length_curve_shape = pm.listRelatives(self.length_curve, shapes=True)[0]
        length_curve_info = CurveInfoNode(name=self.getName("length_info"))
        length_curve_info.input_curve.connect_from(f"{length_curve_shape}.local")
        rest_length = length_curve_info.arc_length.get()
        length_offset = SubtractNode(name=self.getName("length_offset"))
        length_offset.input1.set(rest_length)
        length_offset.input2.connect_from(length_curve_info.arc_length)
        length_offset_blend = MultiplyNode(name=self.getName("length_offset_blend"))
        length_offset_blend.input[0].connect_from(self.preserve_length_att)
        length_offset.output.connect_to(length_offset_blend.input[1])

        length_offset.output.connect_to(f"{self.chest_top_length_adjust}.translateY")

    # =====================================================
    # CONNECTOR
    # =====================================================
    def setRelation(self):
        """Set the relation beetween object from guide to rig"""
        self.relatives["root"] = self.root
        self.relatives["spineBase"] = self.hip_ctl
        self.relatives["hip_pivot"] = self.hip_ctl
        self.relatives["tan0"] = self.mid_ctl
        self.relatives["tan1"] = self.mid_ctl
        self.relatives["spineTop"] = self.top_connection
        self.relatives["chest"] = self.top_connection
        self.relatives["chest_pivot"] = self.chest_ctl

        self.controlRelatives["root"] = self.torso_ctl
        self.controlRelatives["spineBase"] = self.hip_ctl
        self.controlRelatives["hip_pivot"] = self.hip_ctl
        self.controlRelatives["tan0"] = self.mid_ctl
        self.controlRelatives["tan1"] = self.mid_ctl
        self.controlRelatives["spineTop"] = self.chest_top_ctl
        self.controlRelatives["chest"] = self.chest_top_ctl

        self.jointRelatives["root"] = 0
        self.jointRelatives["tan0"] = 0
        self.jointRelatives["spineTop"] = -1
        self.jointRelatives["chest"] = -1
