from dataclasses import dataclass

from maya import cmds

from yrig.control import create_control, set_override_color
from yrig.control.core import Control
from yrig.joint import create_joint
from yrig.skin.split.tag import tag_for_weight_split
from yrig.transform import create_transform

from .guide.core import SplineGuideInfo
from .guide.curve import GuideCurve, get_curve_point_matrix
from .hybrid import build_hybrid
from .module_utils import create_module_switch, module_prep, module_space
from .stabilization import connect_limited_control, create_limited_control
from .utils import align_guides
from .zipper import build_zipper


@dataclass
class ModuleInfo:
    switch_joints: list
    bind_joints: list
    chest_ctrl: Control
    chest_off: Control
    cog_joint: str


class Spine:
    def __init__(
        self,
        zipper_guides: list,
        guides: SplineGuideInfo,
        control_space: list,
        part: str = "spine",
        side: str = "M",
        parent: str = "components",
        control_parent: str | None = None,
        rig_size: float = 1.0,
        joint_parent: str = "skel",
        simple_fk: bool = False,
        simple_ik: bool = False,
        hybrid: bool = True,
        divisions: int = 7,
        root_hook: list | None = None,
        hybrid_mid_shape: str = "round_square",
        chest_des: str = "chest",
        hip_shape: str = "circle",
        hip_des: str = "hip",
        cog_des: str = "cog",
        chest_shape: str = "chest",
        chest_off_shape: str = "hexagon",
        chest_control_size: float = 1 / 3,
        chest_control_pos_offset: tuple = (0, 1 / 8, 0),
        chest_off_control_size: float = 1 / 3.5,
        cog_control_size: float = 1 / 5,
        hip_control_size: float = 1 / 4,
        hyrbid_mid_control_size: float = 1 / 3,
        hybrid_sub_size: float = 1 / 30,
        hybrid_sub_offset: tuple = (0, 0, 1 / 4),
        chest_control_curve_position: float = 0.2,
    ):
        self.part: str = part
        self.side: str = side
        self.parent: str = parent
        self.control_parent: str | None = control_parent
        self.rig_size: float = rig_size
        self.guides = guides
        self.joint_parent = joint_parent
        self.simple_fk = simple_fk
        self.simple_ik = simple_ik
        self.hybrid = hybrid
        self.divisions = divisions + 1
        self.root_hook = root_hook
        if self.side == "M":
            self.main_control_color = "Middle"
            self.sub_control_color = "SubMiddle"
        elif self.side == "L":
            self.main_control_color = "Left"
            self.sub_control_color = "SubLeft"
        else:
            self.main_control_color = "Right"
            self.sub_control_color = "SubRight"
        self.control_space = control_space
        self.hybrid_mid_shape = hybrid_mid_shape
        self.chest_des = chest_des
        self.hip_des = hip_des
        self.cog_des = cog_des
        self.hip_shape = hip_shape
        self.chest_shape = chest_shape
        self.chest_off_shape = chest_off_shape
        self.chest_control_size = chest_control_size
        self.chest_control_pos_offset = chest_control_pos_offset
        self.chest_off_control_size = chest_off_control_size
        self.cog_control_size = cog_control_size
        self.hip_control_size = hip_control_size
        self.hyrbid_mid_control_size = hyrbid_mid_control_size
        self.hybrid_sub_size = hybrid_sub_size
        self.hybrid_sub_offset = hybrid_sub_offset
        self.chest_control_curve_position = chest_control_curve_position
        self.zipper_guides = zipper_guides

    def build(self) -> ModuleInfo:

        # modeule prep work
        prep = module_prep(part=self.part, parent=self.parent, side=self.side, fkik=True, gut=True)
        self.main_grp = prep.main_grp
        self.control_grp = prep.control_grp
        self.guts = prep.guts
        self.ik_control_grp = prep.ik_grp
        self.fk_control_grp = prep.fk_grp

        self.controls = []
        self.bind_joints = []
        split_joints = []
        self.switch_joints = []
        drivers = {}

        self.cog = create_control(
            name=f"{self.cog_des}_{self.side}",
            parent=self.control_grp,
            transform=self.guides.lower.name,
            size=self.rig_size * self.cog_control_size,
            direction="y",
        )

        set_override_color(control=self.cog, color=(0.957, 0.725, 0.259))

        self.hip = create_control(
            name=f"{self.hip_des}_{self.side}",
            parent=self.cog.transform,
            transform=self.guides.lower.name,
            size=self.rig_size * self.hip_control_size,
            direction="y",
        )

        set_override_color(control=self.hip, color=(0.957, 0.725, 0.259))

        if not self.hybrid:
            # Hybrid is currently the only implemented driver system.
            self.hybrid = True

        guide_names = []
        for i in range(self.divisions):
            guide_names.append(f"{self.part}_{self.side}_{i + 1:02d}")
        self.curve = GuideCurve(
            curve=self.guides.curve.name,
            resample_amount=self.divisions,
            output_names=guide_names,
            ignore_handles=True,
            align_normals=False,
            primary_axis="+x",
        )
        for i, guide in enumerate(self.curve.locator_list):
            if guide == self.curve.locator_list[-1]:
                align_guides(guide_01=guide, guide_02=self.curve.locator_list[i - 1], flip_x=True)
            else:
                align_guides(guide_01=guide, guide_02=self.curve.locator_list[i + 1])
        jnt_par = self.guts
        for guide in self.curve.locator_list:
            if guide == self.curve.locator_list[-1]:
                break
            jnt = create_joint(
                name=f"switch_{guide.descriptor}",
                transform=guide.name,
                parent=jnt_par,
                connect=False,
            )
            self.switch_joints.append(jnt)
            jnt_par = jnt

        self.chestswitch_joint = create_joint(
            name=f"switch_{self.chest_des}_{self.side}",
            transform=self.curve.locator_list[-1].name,
            parent=jnt_par,
            connect=False,
        )
        self.switch_joints.append(self.chestswitch_joint)

        # -----------------------------------
        # COG joint - X down chain
        # -----------------------------------

        cog_joint_guide = create_transform(
            name=f"{self.cog_des}_{self.side}_joint_guide",
            transform=self.curve.locator_list[0].name,
            parent=self.guts,
        )

        cog_position = cmds.xform(
            self.cog.transform,
            query=True,
            worldSpace=True,
            translation=True,
        )

        cmds.xform(
            cog_joint_guide,
            worldSpace=True,
            translation=cog_position,  # type:ignore
        )

        self.cog_joint = create_joint(
            name=f"{self.cog_des}_{self.side}_jnt",
            transform=cog_joint_guide,
            parent=self.joint_parent,
            connect=False,
        )

        cmds.parentConstraint(
            self.cog.transform,
            self.cog_joint,
            maintainOffset=True,
        )

        cmds.scaleConstraint(
            self.cog.transform,
            self.cog_joint,
            maintainOffset=True,
        )

        cmds.delete(cog_joint_guide)

        jnt_par = self.cog_joint
        for i, guide in enumerate(self.curve.locator_list):
            if guide == self.curve.locator_list[-1]:
                break
            jnt = create_joint(
                name=f"{guide.descriptor}", transform=guide.name, parent=jnt_par, connect=False
            )
            self.bind_joints.append(jnt)
            split_joints.append(jnt)
            jnt_par = jnt
            cmds.parentConstraint(self.switch_joints[i], jnt)

        tag_for_weight_split(
            influence=self.bind_joints[0],  # <-- your SOURCE joint (must already exist)
            split_influences=split_joints,  # <-- the ones you just created
        )

        self.chest_joint = create_joint(
            name=f"{self.chest_des}_{self.side}",
            transform=self.curve.locator_list[-1].name,
            parent=jnt_par,
            connect=False,
        )
        self.bind_joints.append(self.chest_joint)
        cmds.parentConstraint(self.chestswitch_joint, self.chest_joint)

        # -----------------------------------
        # Main chest control - Y up
        # -----------------------------------

        chest_matrix = get_curve_point_matrix(
            curve=self.guides.curve.name,
            parameter=self.chest_control_curve_position,
            tangent_orient=False,
            aim_axis="y",
        )

        self.chest = create_control(
            name=f"{self.chest_des}_{self.side}",
            parent=self.control_grp,
            transform=chest_matrix,
            size=self.rig_size * self.chest_control_size,
            direction="y",
        )

        set_override_color(
            control=self.chest,
            color=(0.957, 0.725, 0.259),
        )

        for axis in "XYZ":
            cmds.setAttr(
                f"{self.chest.offset}.rotate{axis}",
                0,  # type:ignore
            )

        module_space(
            control=self.chest,
            space_list=[self.cog.transform],
        )

        # -----------------------------------
        # Chest offset control - Y up
        # -----------------------------------

        self.chest_off = create_control(
            name=f"{self.chest_des}_off_{self.side}",
            parent=self.control_grp,
            transform=self.guides.upper.name,
            size=self.rig_size * self.chest_off_control_size,
            direction="y",
        )

        set_override_color(
            control=self.chest_off,
            color=(0.957, 0.725, 0.259),
        )

        # -----------------------------------
        # Hybrid spine
        # -----------------------------------

        if self.hybrid:
            chest_input = create_limited_control(
                control=self.chest_off,
                follow_driver=self.chest.transform,
                parent=self.control_grp,
            )

            hybrid = build_hybrid(
                part=self.part,
                guts=self.guts,
                side=self.side,
                control_grp=self.control_grp,
                guide_curve=self.curve,
                guide_list=self.curve.locator_list,
                cv_transforms=[
                    self.hip.transform,
                    chest_input.raw_driver,
                ],
                rig_size=self.rig_size,
                hip=self.hip,
                hip_des=self.hip_des,
                main_color=self.main_control_color,
                cog=self.cog.transform,
                chest=self.chest.transform,
                mid_control_size=self.hyrbid_mid_control_size,
                sub_spine_control_size=self.hybrid_sub_size,
            )

            # Move the chest offset control to the
            # final limited chest position.

            connect_limited_control(
                control=chest_input,
                limited_driver=hybrid.chest_driver,
                name=f"{self.chest_des}_{self.side}",
            )

            # -----------------------------------
            # Animator squash/stretch attributes
            # -----------------------------------

            for attr in (
                "min_stretch",
                "max_stretch",
                "soft_range",
            ):
                cmds.addAttr(
                    self.chest.transform,
                    longName=attr,
                    proxy=getattr(hybrid, attr),
                    keyable=True,
                )

            # -----------------------------------
            # Hybrid switch drivers
            # -----------------------------------

            drivers["Hybrid"] = {
                "joints": hybrid.joints,
                "visibility": [hybrid.grp],
            }

            # -----------------------------------
            # Zipper controls
            # -----------------------------------

            if self.zipper_guides:
                zipper_grp = create_transform(
                    name=f"{self.part}_{self.side}_zipper_grp",
                    parent=self.control_grp,
                )

                for zipper in self.zipper_guides:
                    build_zipper(
                        guide_curve=zipper,
                        spine_drivers=hybrid.drivers,
                        parent=zipper_grp,
                    )

        # -----------------------------------
        # Spine switch
        # -----------------------------------

        self.spine_switch = create_module_switch(
            driven=self.switch_joints,
            drivers=drivers,
            node_attr=self.chest.transform,
            descriptor=f"{self.part}_{self.side}",
            maintain_offset=False,
        )

        # -----------------------------------
        # Cleanup and module outputs
        # -----------------------------------

        cmds.delete(self.curve.group)

        spine_info = ModuleInfo(
            switch_joints=self.switch_joints,
            bind_joints=self.bind_joints,
            chest_ctrl=self.chest,
            chest_off=self.chest_off,
            cog_joint=self.cog_joint,
        )

        return spine_info
