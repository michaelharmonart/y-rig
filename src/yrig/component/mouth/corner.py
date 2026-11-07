from yrig.control import Control, ControlShape, create_control
from yrig.joint import create_joint
from yrig.maya_api.node import UvPinNode
from yrig.name import Side
from yrig.surface import surface_slide_constraint, uv_pin
from yrig.transform import create_transform, matrix_constraint
from yrig.transform.utils import connect_transform


class MouthCorner:
    def __init__(
        self,
        *,
        side: Side,
        guide: str,
        mouth_surface: str,
        mouth_surface_local: str,
        control_parent: Control | str,
        joint_parent: str,
        parent: str,
        control_size: float = 1,
        def_ribbon: str,
        def_ribbon_uv_pin: UvPinNode,
        mouth_slide_local: str,
    ):
        self.main_control = create_control(
            f"mouth_{side}",
            transform=guide,
            parent=control_parent,
            size=control_size,
            direction="z",
            control_shape=ControlShape.TRIANGLE,
        )
        self.main_local_npo = create_transform(
            f"mouth_{side}_local_npo", transform=self.main_control.transform, parent=parent
        )
        matrix_constraint(mouth_slide_local, self.main_local_npo)
        self.main_local = create_transform(f"mouth_{side}_local", parent=self.main_local_npo)
        connect_transform(self.main_control.transform, self.main_local)

        self.sub_control = create_control(
            f"mouth_{side}_sub",
            transform=guide,
            parent=self.main_control,
            size=control_size * 0.5,
            direction="z",
        )
        surface_slide_constraint(
            mouth_surface,
            driver_transform=self.main_control.transform,
            slider_transform=self.sub_control.offset,
        )

        self.sub_local_npo = create_transform(
            f"mouth_{side}_sub_local_npo",
            transform=guide,
            parent=self.main_local,
        )
        self.sub_local = create_transform(
            f"mouth_{side}_sub_local",
            parent=self.sub_local_npo,
        )
        connect_transform(self.sub_control.transform, self.sub_local)

        surface_slide_constraint(
            mouth_surface_local,
            driver_transform=self.main_local,
            slider_transform=self.sub_local_npo,
        )

        self.main_joint = create_joint(
            name=f"mouth_{side}", transform=self.sub_local, parent=joint_parent, connect=False
        )
        matrix_constraint(
            self.sub_local,
            self.main_joint,
            False,
            use_joint_orient=True,
            rotate=False,
            scale=False,
            shear=False,
        )
        self.def_joint = create_joint(
            name=f"lip_corner_def_{side}", parent=joint_parent, connect=False
        )
        uv_pin(
            def_ribbon,
            self.def_joint,
            (0.75 if side == Side.LEFT else 0.25, 0.5),
            uv_pin_node=def_ribbon_uv_pin,
            drive_rotate=False,
        )
