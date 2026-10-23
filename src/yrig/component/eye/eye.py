from maya import cmds

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.maya_api.node import BlendColorsNode, MultiplyDivideNode, SumNode
from yrig.transform import create_transform
from yrig.transform.constraint import matrix_constraint

from .curve_eyelid import Eyelid
from .eyeball import Eyeball
from .socket import Socket


class Eye:
    def __init__(
        self,
        part: str = "eye",
        side: str = "",
        parent: str = "face_grp",
        control_parent: str = "neck_M0_head_ctl",
        control_size: float = 1.0,
        parent_jnt: str = "face_jnt",
        look_parent: str = "Look_M_ctrl",
    ):
        self.part: str = part
        self.side: str = side
        self.parent: str = parent
        self.control_parent: str = control_parent
        self.control_size: float = control_size
        self.parent_jnt: str = parent_jnt
        self.look_parent: str = look_parent
        self.guides: dict = {
            "root_name": f"eye_root_{side}",
            "center_piv": f"eye_center_{self.side}",
            "aim": f"eye_aim_{self.side}",
            "eyelid_upblink": f"eyelid_upblink_{self.side}",
            "eyelid_upper_rest": f"eyelid_upper_rest_{self.side}",
            "eyelid_blink": f"eyelid_blink_{self.side}",
            "eyelid_lower_rest": f"eyelid_lower_rest_{self.side}",
            "eyelid_downblink": f"eyelid_downblink_{self.side}",
            "socket_upper_curve": f"socket_upper_curve_{self.side}",
            "socket_lower_curve": f"socket_lower_curve_{self.side}",
            "socket_mid_upper": f"socket_upper_{self.side}",
            "socket_mid_lower": f"socket_lower_{self.side}",
            "eyelid_upper_curve": f"eyelid_upper_curve_{self.side}",  # old eyelid curve names
            "eyelid_lower_curve": f"eyelid_lower_curve_{self.side}",  # old eyelid curve names
            "eye_diam": f"eye_circ_{self.side}",
            "iris_diam": f"iris_circ_{self.side}",
            "pupil_diam": f"pupil_circ_{self.side}",
            "sub_blink": [
                f"upper_macro_01_{self.side}",
                f"upper_macro_02_{self.side}",
                f"upper_macro_03_{self.side}",
                f"lower_macro_01_{self.side}",
                f"lower_macro_02_{self.side}",
                f"lower_macro_03_{self.side}",
            ],
        }

    # -------------------
    # Build steps
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

    def setup_structure(self) -> None:
        self.main_grp = create_transform(name=f"eye_{self.side}", parent=self.parent)
        self.component_grp = create_transform(
            name=f"eye_component_{self.side}", parent=self.main_grp
        )
        cmds.hide(self.component_grp)
        self.control_grp = create_transform(
            name=f"eye_control_{self.side}", parent=self.control_parent
        )

    def create_controls(self) -> None:

        eye_radius: float = self.get_nurbs_surface_radius(obj=self.guides["eye_diam"])
        offset_pos = (eye_radius * 2, 0, eye_radius)
        self.main_ctrl = create_control(
            name=self.guides["root_name"],
            parent=self.control_grp,
            transform=self.guides["center_piv"],
            size=self.control_size,
            control_shape="round_square",
            direction="z",
            position_offset=offset_pos,
        )

    def create_joints(self) -> None:
        self.main_jnt = create_joint(
            name=self.guides["root_name"],
            parent=self.parent_jnt,
            transform=self.main_ctrl.transform,
        )

    def build(self) -> None:
        self.setup_structure()
        self.create_controls()
        self.create_joints()

        cmds.addAttr(
            self.main_ctrl.transform,
            longName="Socket",
            attributeType="enum",
            enumName="-------------",
            keyable=True,
        )

        for vis_attr in ["sub_socket"]:
            cmds.addAttr(
                self.main_ctrl.transform,
                longName=vis_attr,
                attributeType="bool",
                defaultValue=False,
                keyable=True,
            )

        cmds.addAttr(
            self.main_ctrl.transform,
            longName="Eyelid_Controls",
            attributeType="enum",
            enumName="-------------",
            keyable=True,
        )

        cmds.addAttr(
            self.main_ctrl.transform,
            longName="eyelidFollow",
            attributeType="double",
            minValue=0.0,
            maxValue=1.0,
            defaultValue=0.1,
            keyable=True,
        )

        """self.eyelid = Eyelid(
            side=self.side,
            guides=self.guides,
            control_size=self.control_size,
            main_ctrl=self.main_ctrl.transform,
            parent=self.main_grp,
            joint_parent=self.main_jnt,
            component_grp=self.component_grp,
            control_grp=self.control_grp,
        )

        self.eyelid.build_blink()"""

        self.eyelid = Eyelid(
            side=self.side,
            guides=self.guides,
            control_size=self.control_size,
            main_ctrl=self.main_ctrl.transform,
            parent=self.main_grp,
            joint_parent=self.main_jnt,
            component_grp=self.component_grp,
            control_grp=self.control_grp,
        )
        self.eyelid.build()

        self.socket = Socket(
            side=self.side,
            guides=self.guides,
            control_size=self.control_size,
            main_ctrl=self.main_ctrl.transform,
            parent=self.main_grp,
            joint_parent=self.main_jnt,
            component_grp=self.component_grp,
            control_grp=self.control_grp,
        )

        self.socket.build_socket()

        self.eyeball = Eyeball(
            side=self.side,
            guides=self.guides,
            control_size=self.control_size,
            main_ctrl=self.main_ctrl.transform,
            parent=self.main_grp,
            joint_parent=self.main_jnt,
            component_grp=self.component_grp,
            control_grp=self.control_grp,
        )

        self.eyeball.build_eyeball()

        self.eye_look_offset_node = BlendColorsNode.create(name=f"look_blend_{self.side}_BC")

        if self.side == "R":
            mirroraim = MultiplyDivideNode.create(name=f"{self.side}_aim_mirror_MD")
            mirroraim.input2.set((-1, -1, -1))
            mirroraim.input1.connect_from(f"{self.eyeball.look_root}.rotate")
            # add_node.input[0].connect_from(f"{self.eyeball.look_root}.rotate{axes}")
        for axes in ["X", "Y", "Z"]:
            add_node = SumNode.create(name=f"look_{axes}_{self.side}_ADL")
            if self.side == "R":
                add_node.input[0].connect_from(f"{mirroraim.output}{axes}")
            else:
                add_node.input[0].connect_from(f"{self.eyeball.look_root}.rotate{axes}")
            add_node.input[1].connect_from(f"{self.eyeball.eye_ctrl}.rotate{axes}")
            if axes == "X":
                color = "R"
            elif axes == "Y":
                color = "G"
            else:
                color = "B"
            add_node.output.connect_to(f"{self.eye_look_offset_node.color1}{color}")

        self.eye_look_offset_node.blender.set(0.1)
        self.eye_look_offset_node.color2.set((0, 0, 0))
        self.eye_look_offset_node.output.connect_to(f"{self.eyelid.look_offset}.rotate")
        self.eye_look_offset_node.blender.connect_from(f"{self.main_ctrl.transform}.eyelidFollow")

        matrix_constraint(
            source_transform=self.look_parent,
            constrain_transform=self.eyeball.look_ctrl.offset,
            keep_offset=True,
        )
