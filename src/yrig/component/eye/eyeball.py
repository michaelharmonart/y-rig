import math

from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.maya_api.node import (
    BlendColorsNode,
    EulerToQuatNode,
    MultiplyDivideNode,
    SumNode,
)
from yrig.skin.split.tag import tag_for_weight_split
from yrig.transform import create_transform, match_location
from yrig.transform.matrix import matrix_constraint
from yrig.transform.utils import get_position


class Eyeball:
    def __init__(
        self,
        guides: dict,
        side: str = "L",
        control_size: float = 1.0,
        main_ctrl: str = "",
        parent: str = "",
        joint_parent: str = "",
        component_grp: str = "",
        control_grp: str = "",
    ) -> None:
        if guides is None:
            guides = {}
        self.side = side
        self.guides = guides
        self.main_ctrl = main_ctrl
        self.control_size = control_size
        self.parent = parent
        self.joint_parent = joint_parent
        self.component_grp = component_grp
        self.control_grp = control_grp

    # -------------------
    # Helper Functions
    # -------------------

    def get_nurbs_surface_radius(self, obj: str) -> float:
        """
        Returns the radius of a NURBS circle.

        Args:
            obj (str): Transform of the NURBS circle

        Returns:
            float: radius
        """

        bbox: list[int | float] = cmds.exactWorldBoundingBox(obj)

        x: int | float = bbox[3] - bbox[0]
        y: int | float = bbox[4] - bbox[1]
        z: int | float = bbox[5] - bbox[2]

        return max(x, y, z) * 0.5

    def compare_radius_and_angle(self, obj_a: str, obj_b: str) -> float:
        """
        Compares two NURBS surfaces and returns:
        - percentage difference (based on radius)
        - estimated angle (based on cosine relationship)

        Args:
            obj_a (str): reference object (true size)
            obj_b (str): compared object (projected/tilted)

        Returns:
            (percent, angle_degrees)
        """

        r1: float = self.get_nurbs_surface_radius(obj_a)
        r2: float = self.get_nurbs_surface_radius(obj_b)

        if r1 == 0:
            raise RuntimeError(f"{obj_a} has zero radius")

        ratio: float = r2 / r1

        # Clamp for safety (floating point issues)
        ratio = max(min(ratio, 1.0), -1.0)

        angle_rad: float = math.acos(ratio)
        angle_deg: float = math.degrees(angle_rad)

        percent: float = ratio * 100.0

        return angle_deg

    def sphere_edge_loop_offsets(self, radius: float, num_loops: int) -> list:
        """
        Returns Y offsets from center loop (equator) to pole on a sphere.
        """
        offsets = []

        for i in range(num_loops + 1):
            t = i / float(num_loops)  # 0 → 1
            theta = t * (math.pi / 2)  # equator → pole
            y = radius * math.sin(theta)
            offsets.append(y)

        return offsets

    def create_eye_preview_circle(self, name_suffix: str, parent: str) -> str:
        eye_radius: float = self.get_nurbs_surface_radius(self.guides["eye_diam"])

        eye_center_pos = get_position(self.guides["center_piv"])

        crv: str = cmds.circle(  # type:ignore
            name=f"preview_{name_suffix}_crv",
            radius=eye_radius,
            normal=[0, 0, 1],  # type:ignore
            sections=16,
            degree=3,
        )[0]

        cmds.parent(crv, parent)
        match_location(transform=crv, target_transform=self.guides["center_piv"])

        return crv

    def dilation_nodes(
        self,
        circle_type: str,
        obj: str,
        eye_radius: float,
        dilation_attr: str,
        new_attr: bool,
        new_attr_tgt: None | str,
        offset_x_attr: str,
        offset_y_attr: str,
        offset_z_attr: str,
        scale_x_attr: str,
        scale_y_attr: str,
    ) -> None:

        if new_attr:
            cmds.addAttr(new_attr_tgt, longName=f"{circle_type}_{obj}_dilation", keyable=True)  # type:ignore
            dilation_attr = f"{new_attr_tgt}.{circle_type}_{obj}_dilation"

        dilation_mult = MultiplyDivideNode.create(
            name=f"{circle_type}_dilation_mult_{self.side}_MD"
        )
        cmds.setAttr(f"{dilation_mult.input2.x}", 18)  # type:ignore  # Convert normalized dilation amount into spherical rotation angle
        dilation_mult.input1.x.connect_from(source_attr=dilation_attr)
        etq_node = EulerToQuatNode.create(
            name=f"{circle_type}_dilation_mult_{self.side}_ETQ",
        )
        etq_node.input_rotate.x.connect_from(f"{dilation_mult.output.x}")
        radius_adjust = MultiplyDivideNode.create(
            name=f"{circle_type}_radius_adjust_{self.side}_MD"
        )
        radius_adjust.input1.x.connect_from(etq_node.output_quat.x)

        z_trans_adl = SumNode.create(name=f"{circle_type}_translateZ_{self.side}_ADL")

        radius_adjust.output.x.connect_to(z_trans_adl.input[0])

        z_trans_adl.output.connect_to(f"{obj}.translateZ")

        # radius_adjust.output.x.connect_to(f"{obj}.translateZ")
        etq_node.output_quat.w.connect_to(f"{obj}.scaleZ")
        radius_adjust.input2.x.set(eye_radius)

        ## offsets

        radius_adjust.input1.y.connect_from(etq_node.output_quat.w)
        radius_adjust.input1.z.connect_from(etq_node.output_quat.w)
        radius_adjust.input2.z.connect_from(scale_x_attr)
        radius_adjust.input2.y.connect_from(scale_y_attr)
        radius_adjust.output.y.connect_to(f"{obj}.scaleY")
        radius_adjust.output.z.connect_to(f"{obj}.scaleX")

        z_trans_adl.input[1].connect_from(offset_z_attr)
        cmds.connectAttr(offset_y_attr, f"{obj}.translateY")
        cmds.connectAttr(offset_x_attr, f"{obj}.translateX")

    def build_eyeball(self) -> None:
        eye_radius: float = self.get_nurbs_surface_radius(self.guides["eye_diam"])
        cmds.addAttr(
            self.main_ctrl,
            longName="dilation_controls",
            attributeType="enum",
            enumName="-------------",
            keyable=True,
        )

        self.eye_ctrl = create_control(
            name=f"eye_{self.side}",
            parent=self.main_ctrl,
            transform=self.guides["center_piv"],
            control_shape="circle",
            direction="z",
            size=eye_radius / 4,
            position_offset=(0, 0, eye_radius),
        )
        self.eye_jnt = create_joint(
            transform=self.eye_ctrl.transform, name=f"eye_{self.side}", parent=self.joint_parent
        )

        self.look_ctrl = create_control(
            name=f"look_{self.side}",
            parent=self.control_grp,
            transform=self.guides["aim"],
            control_shape="circle",
            direction="z",
            size=eye_radius / 2,
        )

        self.look_root = create_joint(
            transform=self.eye_ctrl.transform,
            name=f"look_root_{self.side}",
            parent=self.component_grp,
            connect=False,
        )

        self.look_end = create_joint(
            transform=self.look_ctrl.transform,
            name=f"look_end_{self.side}",
            parent=self.look_root,
            connect=False,
        )

        self.ik_handle, self.effector = cmds.ikHandle(
            name=f"aim_sc_{self.side}_ikh",
            startJoint=self.look_root,
            endEffector=self.look_end,
            solver="ikSCsolver",
        )  # type: ignore

        cmds.parent(self.ik_handle, self.component_grp)

        matrix_constraint(self.look_ctrl.transform, self.ik_handle)
        cmds.pointConstraint(self.main_ctrl, self.look_root, maintainOffset=True)
        cmds.orientConstraint(self.look_root, self.eye_ctrl.offset, maintainOffset=True)

        pupil_degree: float = round(
            self.compare_radius_and_angle(self.guides["eye_diam"], self.guides["pupil_diam"])
        )

        iris_degree: float = round(
            self.compare_radius_and_angle(self.guides["eye_diam"], self.guides["iris_diam"])
        )

        pupil_percent = pupil_degree / 90
        iris_percent = iris_degree / 90

        eye_center_pos = get_position(self.guides["center_piv"])

        percents = [0, iris_percent, pupil_percent, 1]

        cmds.addAttr(
            self.main_ctrl,
            longName="pupil_dilation",
            attributeType="double",
            minValue=0 - (pupil_percent * 10),
            maxValue=10 - (pupil_percent * 10),
            keyable=True,
        )
        cmds.addAttr(
            self.main_ctrl,
            longName="iris_dilation",
            attributeType="double",
            minValue=0 - (iris_percent * 10),
            maxValue=10 - (iris_percent * 10),
            keyable=True,
        )

        cmds.addAttr(
            self.main_ctrl,
            longName="end_dilation",
            attributeType="double",
            minValue=0,
            maxValue=10,
            defaultValue=10,
            keyable=False,
        )

        cmds.addAttr(
            self.main_ctrl,
            longName="center_dilation",
            attributeType="double",
            minValue=0,
            maxValue=10,
            defaultValue=0,
            keyable=False,
        )

        dilation_offset = create_transform(
            name=f"dilation_{self.side}_Offset",
            parent=self.eye_ctrl.transform,
            transform=self.guides["center_piv"],
        )

        cmds.hide(dilation_offset)

        self.preview_circles = {}

        for i, circle_type in enumerate(["center", "iris", "pupil", "end"]):
            circle = self.create_eye_preview_circle(
                name_suffix=f"{circle_type}_{self.side}", parent=f"{dilation_offset}"
            )
            self.preview_circles[f"{circle_type}"] = circle

            cmds.addAttr(circle, longName="dilation_amount", attributeType="double", keyable=True)

            key = circle_type in ["iris", "pupil"]

            cmds.addAttr(
                self.main_ctrl,
                longName=f"{circle_type}_adjustments",
                attributeType="enum",
                enumName="----",
                keyable=key,
            )

            cmds.addAttr(
                self.main_ctrl,
                longName=f"{circle_type}_scaleX",
                attributeType="double",
                defaultValue=1,
                keyable=key,
            )

            cmds.addAttr(
                self.main_ctrl,
                longName=f"{circle_type}_scaleY",
                attributeType="double",
                defaultValue=1,
                keyable=key,
            )

            cmds.addAttr(
                self.main_ctrl,
                longName=f"{circle_type}_offsetY",
                attributeType="double",
                defaultValue=0,
                keyable=key,
            )

            cmds.addAttr(
                self.main_ctrl,
                longName=f"{circle_type}_offsetX",
                attributeType="double",
                defaultValue=0,
                keyable=key,
            )

            cmds.addAttr(
                self.main_ctrl,
                longName=f"{circle_type}_offsetZ",
                attributeType="double",
                defaultValue=0,
                keyable=key,
            )

            if circle_type == "center":
                pass
            else:
                # percents[i]
                dilation_offset_node = SumNode.create(
                    name=f"{circle_type}_dilation_mult_{self.side}_ADL"
                )
                cmds.connectAttr(
                    f"{self.main_ctrl}.{circle_type}_dilation", f"{dilation_offset_node.input[1]}"
                )
                cmds.connectAttr(f"{dilation_offset_node.output}", f"{circle}.dilation_amount")
                cmds.setAttr(f"{dilation_offset_node.input[0]}", percents[i] * 10)  # type:ignore

                self.dilation_nodes(
                    circle_type=circle_type,
                    obj=circle,
                    eye_radius=eye_radius,
                    dilation_attr=f"{dilation_offset_node.output}",
                    new_attr=False,
                    new_attr_tgt=None,
                    scale_x_attr=f"{self.main_ctrl}.{circle_type}_scaleX",
                    scale_y_attr=f"{self.main_ctrl}.{circle_type}_scaleY",
                    offset_x_attr=f"{self.main_ctrl}.{circle_type}_offsetX",
                    offset_y_attr=f"{self.main_ctrl}.{circle_type}_offsetY",
                    offset_z_attr=f"{self.main_ctrl}.{circle_type}_offsetZ",
                )

        # joints
        loop_num = 10
        loops_list = self.sphere_edge_loop_offsets(radius=eye_radius, num_loops=loop_num)

        self.dilation_joints = []

        for i, _loop in enumerate(loops_list):
            parent = self.eye_jnt

            jnt = create_joint(
                name=f"eye_dilation_{i:02d}_{self.side}",
                parent=parent,
                transform=self.guides["center_piv"],
                connect=False,
            )
            self.dilation_joints.append(jnt)
            cmds.setAttr(f"{jnt}.translateZ", _loop)

            x = i / 10.0
            if x < iris_percent:
                blend = ["iris", "center"]
                blend_num = x / iris_percent

            elif x < pupil_percent:
                blend = ["pupil", "iris"]
                blend_num = (x - iris_percent) / (pupil_percent - iris_percent)

            else:
                blend = ["pupil", "pupil"]
                blend_num = (x - pupil_percent) / (1.0 - pupil_percent)

            blendnode = BlendColorsNode.create(name=f"blend_dilation_{i:02d}_{self.side}_BC")
            blendnode2 = BlendColorsNode.create(name=f"blend_offset_{i:02d}_{self.side}_BC")
            blendnode.color1.r.connect_from(
                source_attr=f"{self.preview_circles[blend[0]]}.dilation_amount"
            )
            blendnode.color2.r.connect_from(
                source_attr=f"{self.preview_circles[blend[1]]}.dilation_amount"
            )
            blendnode.blender.set(blend_num)

            blendnode.color1.g.connect_from(source_attr=f"{self.main_ctrl}.{blend[0]}_scaleY")
            blendnode.color1.b.connect_from(source_attr=f"{self.main_ctrl}.{blend[0]}_scaleX")
            blendnode.color2.g.connect_from(source_attr=f"{self.main_ctrl}.{blend[1]}_scaleY")
            blendnode.color2.b.connect_from(source_attr=f"{self.main_ctrl}.{blend[1]}_scaleX")

            blendnode2.blender.set(blend_num)

            blendnode2.color1.g.connect_from(source_attr=f"{self.main_ctrl}.{blend[0]}_offsetY")
            blendnode2.color1.b.connect_from(source_attr=f"{self.main_ctrl}.{blend[0]}_offsetX")
            blendnode2.color2.g.connect_from(source_attr=f"{self.main_ctrl}.{blend[1]}_offsetY")
            blendnode2.color2.b.connect_from(source_attr=f"{self.main_ctrl}.{blend[1]}_offsetX")
            blendnode2.color1.r.connect_from(source_attr=f"{self.main_ctrl}.{blend[0]}_offsetZ")
            blendnode2.color2.r.connect_from(source_attr=f"{self.main_ctrl}.{blend[1]}_offsetZ")

            self.dilation_nodes(
                circle_type=f"loop{i:02d}",
                obj=jnt,
                eye_radius=eye_radius,
                dilation_attr=f"{blendnode.output.r}",
                new_attr=False,
                new_attr_tgt=None,
                scale_x_attr=f"{blendnode.output.b}",
                scale_y_attr=f"{blendnode.output.g}",
                offset_x_attr=f"{blendnode2.output.b}",
                offset_y_attr=f"{blendnode2.output.g}",
                offset_z_attr=f"{blendnode2.output.r}",
            )
        tag_for_weight_split(
            influence=self.dilation_joints[0],
            split_influences=self.dilation_joints,
        )

        # proxy attrs

        for attr in [
            "dilation_controls",
            "iris_dilation",
            "pupil_dilation",
            "iris_adjustments",
            "iris_offsetX",
            "iris_offsetY",
            "iris_offsetZ",
            "iris_scaleX",
            "iris_scaleY",
            "pupil_adjustments",
            "pupil_offsetX",
            "pupil_offsetY",
            "pupil_offsetZ",
            "pupil_scaleX",
            "pupil_scaleY",
        ]:
            for ctrl in [self.eye_ctrl.transform, self.look_ctrl.transform]:
                cmds.addAttr(ctrl, longName=attr, proxy=f"{self.main_ctrl}.{attr}")
