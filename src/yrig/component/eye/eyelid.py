import math
from dataclasses import dataclass
from typing import Any, Literal

from maya import cmds
from maya.api.OpenMaya import MEulerRotation, MMatrix, MSpace, MTransformationMatrix, MVector

from yrig.control import ControlShape, create_control
from yrig.control.core import Control
from yrig.joint import create_joint
from yrig.maya_api.node import (
    ConditionNode,
    DecomposeMatrixNode,
    MultiplyDivideNode,
    MultMatrixNode,
    PlusMinusAverageNode,
    SumNode,
)
from yrig.skin.split.tag import tag_for_weight_split
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.transform import create_transform
from yrig.transform.matrix import matrix_constraint
from yrig.transform.utils import get_position

from .guide_curve import GuideCurve


@dataclass
class BlinkControl:
    """
    Stores information about a generated guide locator.
    """

    blink_transform: str
    blink_offset: str
    eyelid_transform: str
    eyelid_top: str
    blink_top: str
    driver_top: str
    driver_driven: str
    driver_driver: str
    joint: str


class Eyelid:
    def __init__(
        self,
        side: str,
        guides: dict,
        main_ctrl: str,
        parent: str,
        joint_parent: str,
        component_grp: str,
        control_grp: str,
        control_size: float = 1.0,
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
    def convert_to_matrix(
        self,
        pos: tuple[float, float, float] = (0, 0, 0),
        rot: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
    ) -> MMatrix:
        """
        Build an MMatrix from translation, rotation, and scale.
        """

        m = MTransformationMatrix()

        # Translation
        m.setTranslation(MVector(*pos), MSpace.kWorld)

        # Rotation (Euler degrees → radians internally handled by API)
        euler = MEulerRotation(
            math.radians(rot[0]),
            math.radians(rot[1]),
            math.radians(rot[2]),
        )
        m.setRotation(euler)

        # Scale
        m.setScale(scale, MSpace.kWorld)

        return m.asMatrix()

    def curve_to_matrix_spline(
        self,
        parent: str,
        curve: str,
        descriptor: str,
        driver_list: list,
        rebuild: bool = False,
        cv_count: int = 10,
        ignore_handles: bool = False,
    ) -> str:
        """
        Returns worldspace positions of CVs on a curve.

        Args:
            curve (str): Name of the curve transform or shape.
            rebuild (bool): If True, duplicate and rebuild curve.
            cv_count (int): Number of CVs if rebuilding.
            ignore_handles (bool): If True, skip 2nd and 2nd-to-last CV.

        Returns:
            list of tuples: [(x, y, z), ...]
        """

        temp_curve = None
        working_curve = curve

        top_grp = create_transform(name=f"{descriptor}_spline_{self.side}_grp", parent=parent)

        # Ensure we are working with the shape node
        shapes = cmds.listRelatives(curve, shapes=True, fullPath=True) or []
        if shapes:
            working_curve = shapes[0]

        # Optional rebuild
        if rebuild:
            temp_curve = cmds.duplicate(curve, name=curve + "_tempRebuild")[0]

            cmds.rebuildCurve(
                temp_curve,
                ch=False,  # type:ignore
                rpo=True,  # type:ignore
                rt=0,  # type:ignore
                end=1,  # type:ignore
                kr=0,  # type:ignore
                kcp=False,  # type:ignore
                kep=True,  # type:ignore
                kt=False,  # type:ignore
                s=cv_count - 1,  # type:ignore
                d=3,  # type:ignore
            )

            # Get shape of rebuilt curve
            shapes = cmds.listRelatives(temp_curve, shapes=True, fullPath=True) or []
            if shapes:
                working_curve = shapes[0]
            else:
                working_curve = temp_curve

        # Get CV count
        spans = cmds.getAttr(working_curve + ".spans")
        degree = cmds.getAttr(working_curve + ".degree")
        cv_total = spans + degree

        indices = list(range(cv_total))

        # Ignore handles if requested
        if ignore_handles and cv_total > 3:
            indices = [i for i in indices if i not in (1, cv_total - 2)]

        self.sub_eyelid_controls = []
        self.sub_eyelid_joints = []
        sub_eyelid_offsets = []
        for i in indices:  # descriptor
            cv = f"{working_curve}.cv[{i}]"

            # Get CV position
            pos = get_position(cv)

            # Create temp transform
            temp = cmds.group(empty=True, name=f"{curve}_tempCv_{i}#")
            cmds.xform(temp, worldSpace=True, translation=(pos.x, pos.y, pos.z))

            sub_ctrl = create_control(
                name=f"{descriptor}_{i}_{self.side}",
                parent=top_grp,
                transform=temp,
                size=self.control_size / 10,
                control_shape="circle",
                direction="z",
            )
            sub_jnt = create_joint(
                name=f"{descriptor}_{i}_{self.side}",
                parent=self.joint_parent,
                transform=sub_ctrl.transform,
            )

            self.sub_eyelid_controls.append(sub_ctrl)
            self.sub_eyelid_joints.append(sub_jnt)
            sub_eyelid_offsets.append(sub_ctrl.offset)

            cmds.delete(temp)

        # Cleanup
        if temp_curve and cmds.objExists(temp_curve):
            cmds.delete(temp_curve)

        tag_for_weight_split(
            influence=self.sub_eyelid_joints[0],  # <-- your SOURCE joint (must already exist)
            split_influences=self.sub_eyelid_joints,  # <-- the ones you just created
        )

        matrix_spline_from_transforms(
            name=f"{self.side}_{descriptor}",
            pinned_transforms=sub_eyelid_offsets,
            cv_transforms=driver_list,
            parent=self.component_grp,
            degree=2,
        )

        return top_grp

    def get_flat_y_aim_rotation(self, source: str, target: str) -> float:
        """
        Returns Y-axis rotation (degrees) from source → target,
        ignoring Y height (XZ plane only).
        """

        p1: Any = get_position(transform=source)
        p2: Any = get_position(transform=target)

        # Flatten Y
        dx = p2.x - p1.x
        dz = p2.z - p1.z

        # Angle in radians
        angle = math.atan2(dx, dz)

        # Convert to degrees
        return math.degrees(angle)

    def soft_colide(
        self,
        upper_driver: str,
        lower_driver: str,
        upper_driven: str,
        lower_driven: str,
        parent: str,
        push: float = 0.5,
        rot_mult: float = -50,
    ) -> None:

        ctrl_list = [lower_driver, upper_driver]

        out_matrix = []

        # shared logic to check how close our two drivers are

        pma_calc = PlusMinusAverageNode.create(name=f"{upper_driver}_{lower_driver}_PMA")
        pma_calc.operation.set(2)

        condition = ConditionNode.create(name=f"{upper_driver}_{lower_driver}_COND")
        condition.operation.set(2)
        condition.color_if_false.set((0, 0, 0))

        pma_calc.output_1d.connect_to(condition.color_if_true.r)
        pma_calc.output_1d.connect_to(condition.first_term)

        # cmds.connectAttr(f"{pma_calc}.output1D", f"{condition}.colorIfTrueR")
        # cmds.connectAttr(f"{pma_calc}.output1D", f"{condition}.firstTerm")

        rot_md = MultiplyDivideNode.create(name=f"{upper_driver}_{lower_driver}_MD")

        for ctrl in ctrl_list:
            cmds.addAttr(ctrl, longName="push", at="double", dv=push, k=True)  # type:ignore
            cmds.addAttr(ctrl, longName="rot_mult_DEV", at="double", dv=rot_mult, k=True)  # type:ignore
            mult_matrix = MultMatrixNode.create(name=f"{ctrl}_{self.side}_MM")
            dec_matrix = DecomposeMatrixNode.create(name=f"{ctrl}_{self.side}_DM")

            # Connect world matrix → multMatrix
            mult_matrix.matrix_in[0].connect_from(f"{ctrl}.worldMatrix[0]")
            mult_matrix.matrix_in[1].connect_from(f"{parent}.worldInverseMatrix[0]")

            # multMatrix → decomposeMatrix
            dec_matrix.input_matrix.connect_from(mult_matrix.matrix_sum)

            # Connect output Y into PMA
            if ctrl == ctrl_list[0]:
                dec_matrix.output_translate.y.connect_to(pma_calc.input_1d[0])
                cmds.connectAttr(f"{ctrl}.rot_mult_DEV", f"{rot_md.input2.x}")
                cmds.connectAttr(f"{rot_md.output.x}", f"{lower_driven}.rotateX")

            else:
                dec_matrix.output_translate.y.connect_to(pma_calc.input_1d[1])
                cmds.connectAttr(f"{ctrl}.rot_mult_DEV", f"{rot_md.input2.y}")
                cmds.connectAttr(f"{rot_md.output.y}", f"{upper_driven}.rotateX")

            out_matrix.append(dec_matrix)

            # push logic

            # -------------------------
            # push multiplyDivide
            # -------------------------
            push_mult = MultiplyDivideNode.create(name=f"{ctrl}_{self.side}_MD")

            push_mult.input2.x.set(0.5 if ctrl == ctrl_list[0] else -0.5)
            push_mult.operation.set(1)  # assuming multiply

            condition.out_color.r.connect_to(push_mult.input1.x)

            # -------------------------
            # PlusMinusAverageNode driver
            # -------------------------
            pma_drive = PlusMinusAverageNode.create(name=f"{ctrl}_{self.side}_PMA")
            pma_drive.operation.set(2)

            dec_matrix.output_translate.y.connect_to(pma_drive.input_1d[0])
            push_mult.output.x.connect_to(pma_drive.input_1d[1])

            if ctrl == ctrl_list[0]:
                cmds.connectAttr(f"{pma_drive.output_1d}", f"{rot_md.input1.x}")
            else:
                cmds.connectAttr(f"{pma_drive.output_1d}", f"{rot_md.input1.y}")

    def build_blink(
        self, z_offset: float = 1, x_offset: float = 1, colide_offset: float = 2
    ) -> None:

        self.sub_eyelid_controls = []

        self.upper_guides = GuideCurve(
            curve=self.guides["eyelid_upper_curve"],
            resample_amount=7,
            output_names=[
                "upper_inner_corner",
                "upper_inner_01",
                "upper_inner_02",
                "upper_mid",
                "upper_outer_02",
                "upper_outer_01",
                "upper_outer_corner",
            ],
            ignore_handles=True,
            align_normals=True,
        )

        self.lower_guides = GuideCurve(
            curve=self.guides["eyelid_lower_curve"],
            resample_amount=7,
            output_names=[
                "lower_inner_corner",
                "lower_inner_01",
                "lower_inner_02",
                "lower_mid",
                "lower_outer_02",
                "lower_outer_01",
                "lower_outer_corner",
            ],
            ignore_handles=True,
            align_normals=True,
        )

        pos_list = []

        upper_pos: Any = get_position(transform=self.upper_guides.locator_list[3].name)
        lower_pos: Any = get_position(transform=self.lower_guides.locator_list[3].name)

        pos_list.append(upper_pos)
        pos_list.append(lower_pos)

        blink_x: float = (upper_pos.x + lower_pos.x) / 2
        blink_y: float = (upper_pos.y + lower_pos.y) / 2
        blink_z: float = (upper_pos.z + lower_pos.z) / 2 + z_offset

        self.main_blink_controls = []
        self.main_eyelid_controls: dict[str, BlinkControl] = {}
        self.sub_blink_offsets = []

        # BlinkControl

        #######
        # Set up look follow
        #######

        self.look_offset = create_transform(
            name=f"look_offset_{self.side}",
            parent=self.main_ctrl,
            transform=self.guides["center_piv"],
        )

        #######
        # Sub Blink Set up // Main Eyelid Set up
        #######
        twist_grps = []
        for side in ("upper", "lower"):
            sub_transform_grp = create_transform(
                name=f"{side}_sub_blink_offset_matrix_drivers_{self.side}",
                parent=self.look_offset,
                transform=self.guides["center_piv"],
            )
            twist_grps.append(sub_transform_grp)

        sub_mult = [-40, -40, -40, -40, -40, -40, -40]

        for x, sub_blink in enumerate(
            [
                "inner_corner",
                "inner_01",
                "inner_02",
                "mid",
                "outer_02",
                "outer_01",
                "outer_corner",
            ]
        ):
            driven_grps_list: list[str] = []
            driver_grps_list: list[str] = []

            p1 = get_position(transform=self.upper_guides.locator_list[3].name)
            p2 = get_position(transform=self.lower_guides.locator_list[3].name)

            y_offset = (MVector(p2.x, p2.y, p2.z) - MVector(p1.x, p1.y, p1.z)).length() / 10
            for i, side in enumerate(["upper", "lower"]):
                # setting up mods

                up_mod: Literal[1, -1] = 1 if side == "upper" else -1

                side_mod: Literal[1, -1] = 1 if self.side == "L" else -1

                if sub_blink in ["inner_corner", "inner_01", "inner_02"]:
                    mod = -0.5
                elif sub_blink == "mid":
                    mod = 0
                else:
                    mod = 0.5

                if sub_blink in ["inner_01", "outer_01"]:
                    mod = mod * 1.5
                elif sub_blink in ["inner_corner", "outer_corner"]:
                    mod = mod * 2

                new_matrix: Any = self.convert_to_matrix(
                    pos=(
                        blink_x + (mod * side_mod * x_offset),
                        blink_y + (up_mod * y_offset * colide_offset),
                        blink_z,
                    )
                )

                # building sub controls

                blink_control: Control = create_control(
                    name=f"{sub_blink}_{side}_blink_{self.side}",
                    parent=self.main_ctrl,
                    transform=new_matrix,
                    size=self.control_size / 10,
                    control_shape="sphere",
                    direction="z",
                )

                sdk_grp = create_transform(
                    name=f"{sub_blink}_{side}_blink_{self.side}_SDK",
                    parent=blink_control.offset,
                    transform=new_matrix,
                )

                self.sub_blink_offsets.append(blink_control.offset)

                cmds.parent(
                    blink_control.transform,
                    sdk_grp,
                )
                # setting up blink driver groups

                driver_offset: str = create_transform(
                    name=f"{sub_blink}_{side}_blink_offset_{self.side}",
                    parent=twist_grps[i],
                    transform=self.guides["center_piv"],
                )

                driver_driven: str = create_transform(
                    name=f"{sub_blink}_{side}_blink_driven_{self.side}",
                    parent=driver_offset,
                    transform=self.guides["center_piv"],
                )

                driven_grps_list.append(driver_driven)

                driver_driver: str = create_transform(
                    name=f"{sub_blink}_{side}_blink_driver_{self.side}",
                    parent=driver_driven,
                    transform=self.guides["center_piv"],
                )

                driver_grps_list.append(driver_driver)

                if side == "upper":
                    guide = self.upper_guides.locator_list[x].name
                else:
                    guide = self.lower_guides.locator_list[x].name

                aim: float = self.get_flat_y_aim_rotation(
                    source=self.guides["center_piv"],
                    target=guide,
                )
                counter_aim = cmds.getAttr(f"{self.guides['center_piv']}.rotateY")

                # cmds.setAttr(f"{driver_offset}.rotateY", aim - counter_aim)

                eyelid_control = create_control(
                    name=f"{sub_blink}_{side}_eyelid_{self.side}",
                    parent=self.main_ctrl,
                    transform=guide,
                    size=self.control_size / 4,
                    control_shape="circle",
                    direction="z",
                )

                self.sub_eyelid_controls.append(eyelid_control)

                joint = create_joint(
                    name=f"{sub_blink}_{side}_eyelid_{self.side}",
                    transform=eyelid_control.transform,
                    parent=self.joint_parent,
                )

                self.main_eyelid_controls[f"{side}_{sub_blink}"] = BlinkControl(
                    blink_top=blink_control.offset,
                    blink_transform=blink_control.transform,
                    eyelid_transform=eyelid_control.transform,
                    blink_offset=sdk_grp,
                    eyelid_top=eyelid_control.offset,
                    driver_top=driver_offset,
                    driver_driven=driver_driven,
                    driver_driver=driver_driver,
                    joint=joint,
                )

                ##### Adding x translate control funtionality to the controls

                x_md = MultiplyDivideNode.create(name=f"{sub_blink}_{side}_{self.side}_MD")

                x_md.input1.x.connect_from(
                    f"{self.main_eyelid_controls[f'{side}_{sub_blink}'].blink_transform}.translateX"
                )
                x_md.output.x.connect_to(f"{driver_driven}.rotateZ")
                x_md.input2.x.set(-30)

            self.soft_colide(
                upper_driver=self.main_eyelid_controls[f"upper_{sub_blink}"].blink_transform,
                lower_driver=self.main_eyelid_controls[f"lower_{sub_blink}"].blink_transform,
                upper_driven=driven_grps_list[0],
                lower_driven=driven_grps_list[1],
                parent=self.main_ctrl,
                push=0.5,
                rot_mult=sub_mult[x],
            )

            if sub_blink in ["inner_corner", "outer_corner"]:
                continue
            else:
                matrix_constraint(
                    source_transform=driver_grps_list[0],
                    constrain_transform=self.main_eyelid_controls[f"upper_{sub_blink}"].eyelid_top,
                    keep_offset=True,
                )
                matrix_constraint(
                    driver_grps_list[1],
                    self.main_eyelid_controls[f"lower_{sub_blink}"].eyelid_top,
                    keep_offset=True,
                )
        for sub_blink in ["inner", "outer"]:
            new_driver_list = [
                self.main_eyelid_controls[f"upper_{sub_blink}_corner"].driver_driver,
                self.main_eyelid_controls[f"lower_{sub_blink}_corner"].driver_driver,
            ]
            for i, side in enumerate(["upper", "lower"]):
                matrix_constraint(
                    source_transform=new_driver_list[i],
                    constrain_transform=self.main_eyelid_controls[
                        f"{side}_{sub_blink}_corner"
                    ].eyelid_top,
                    translate=False,
                    rotate=True,
                    keep_offset=True,
                    scale=False,
                    shear=False,
                )
                cmds.pointConstraint(
                    self.main_eyelid_controls[f"{side}_{sub_blink}_01"].eyelid_transform,
                    self.main_eyelid_controls[f"{side}_{sub_blink}_corner"].eyelid_top,
                    maintainOffset=True,
                )
                cmds.pointConstraint(
                    self.main_ctrl,
                    self.main_eyelid_controls[f"{side}_{sub_blink}_corner"].eyelid_top,
                    maintainOffset=True,
                )

        ##########
        # Main Control Behavior
        ##########
        for i, side in enumerate(["upper", "lower"]):
            side_mod = -1 if self.side == "R" else 1
            blink_matrix: Any = self.convert_to_matrix(
                pos=(blink_x, blink_y, blink_z), scale=(1 * side_mod, 1, 1)
            )
            blink_ctrl = create_control(
                name=f"{side}_blink_{self.side}",
                parent=self.main_ctrl,
                transform=blink_matrix,
                size=self.control_size,
                control_shape=ControlShape.SEMI_CIRCLE,
                direction="z",
                dimensions=(1, 1, 1 if side == "upper" else -1),
            )
            twist_md = MultiplyDivideNode.create(name=f"{self.side}_{side}_twist_DM")
            cmds.connectAttr(f"{blink_ctrl.transform}.translateX", f"{twist_md.input1.x}")
            cmds.setAttr(f"{twist_md.input2.x}", 15)  # type:ignore

            self.main_blink_controls.append(blink_ctrl)
            cmds.connectAttr(f"{twist_md.output.x}", f"{twist_grps[i]}.rotateY")

            cmds.connectAttr(
                f"{blink_ctrl.transform}.translateY",
                f"{self.main_eyelid_controls[f'{side}_mid'].blink_offset}.translateY",
            )
            mod_values = [0.5, -1.5, -1, 1, 1.5, 0.5]
            for i, sub in enumerate(
                ["inner_corner", "inner_01", "inner_02", "outer_02", "outer_01", "outer_corner"]
            ):
                mod: float = mod_values[i]
                input_mult = MultiplyDivideNode.create(name=f"{self.side}_{side}_{sub}_input_MD")
                sum_node: SumNode = SumNode.create(name=f"{self.side}_{side}_{sub}_sum")
                input_mult.input1.x.connect_from(f"{blink_ctrl.transform}.rotateZ")
                cmds.setAttr(f"{input_mult.input2.x}", 0.03 * mod)  # type:ignore
                cmds.connectAttr(f"{blink_ctrl.transform}.translateY", f"{sum_node.input[0]}")
                cmds.connectAttr(f"{input_mult.output.x}", f"{sum_node.input[1]}")
                cmds.connectAttr(
                    f"{sum_node.output}",
                    f"{self.main_eyelid_controls[f'{side}_{sub}'].blink_offset}.translateY",
                )

        self.upper_driver_joint = [
            self.main_eyelid_controls["upper_inner_corner"].joint,
            self.main_eyelid_controls["upper_inner_01"].joint,
            self.main_eyelid_controls["upper_inner_02"].joint,
            self.main_eyelid_controls["upper_mid"].joint,
            self.main_eyelid_controls["upper_outer_02"].joint,
            self.main_eyelid_controls["upper_outer_01"].joint,
            self.main_eyelid_controls["upper_outer_corner"].joint,
        ]
        self.lower_driver_joint = [
            self.main_eyelid_controls["lower_inner_corner"].joint,
            self.main_eyelid_controls["lower_inner_01"].joint,
            self.main_eyelid_controls["lower_inner_02"].joint,
            self.main_eyelid_controls["lower_mid"].joint,
            self.main_eyelid_controls["lower_outer_02"].joint,
            self.main_eyelid_controls["lower_outer_01"].joint,
            self.main_eyelid_controls["lower_outer_corner"].joint,
        ]

        tag_for_weight_split(
            influence=self.lower_driver_joint[0],  # <-- your SOURCE joint (must already exist)
            split_influences=self.lower_driver_joint,  # <-- the ones you just created
        )

        tag_for_weight_split(
            influence=self.upper_driver_joint[0],  # <-- your SOURCE joint (must already exist)
            split_influences=self.upper_driver_joint,  # <-- the ones you just created
        )

        cmds.addAttr(
            self.main_ctrl,
            longName="eyelid_controls",
            attributeType="enum",
            enumName="-------------",
            keyable=True,
        )

        for vis_attr in ["sub_blink", "sub_eyelid", "sub_socket"]:
            cmds.addAttr(
                self.main_ctrl,
                longName=vis_attr,
                attributeType="bool",
                defaultValue=False,
                keyable=True,
            )

            for control in self.main_blink_controls:
                cmds.addAttr(
                    f"{control.transform}", longName=vis_attr, proxy=f"{self.main_ctrl}.{vis_attr}"
                )

        for control in self.sub_blink_offsets:
            cmds.connectAttr(f"{self.main_ctrl}.sub_blink", f"{control}.visibility")
            cmds.addAttr(
                f"{control}",
                longName="sub_blink",
                proxy=f"{self.main_ctrl}.sub_blink",
            )
        for control in self.sub_eyelid_controls:
            cmds.connectAttr(f"{self.main_ctrl}.sub_eyelid", f"{control.transform}.visibility")
            cmds.addAttr(
                f"{control.transform}",
                longName="sub_eyelid",
                proxy=f"{self.main_ctrl}.sub_eyelid",
            )

        # end of blink
        cmds.delete(self.upper_guides.group, self.lower_guides.group)
