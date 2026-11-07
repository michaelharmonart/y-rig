from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from maya import cmds

from yrig.control import Control, ControlShape, create_control
from yrig.deformer.blendshape import create_blendshape
from yrig.deformer.blendshape.core import add_blendshape_target_from_geometry
from yrig.name import Side
from yrig.skin import skin_and_apply_weights
from yrig.surface import setup_uv_pin_node, surface_slide_constraint
from yrig.transform import create_transform, matrix_constraint
from yrig.transform.constraint import local_constraint
from yrig.transform.utils import connect_transform

from .corner import MouthCorner
from .lip import Lip, LipGuides, _default_lip_guides


@dataclass
class RibbonLayer:
    name: str
    skin: Path | None


@dataclass
class MouthGuides:
    mouth: str = "mouth_M"
    mouth_surface: str = "face_surface"
    lip_ribbon: str = "lip_ribbon"
    left_corner: str = "mouth_L"
    right_corner: str = "mouth_R"
    upper_lip: LipGuides = field(default_factory=lambda: _default_lip_guides(side="upper"))
    lower_lip: LipGuides = field(default_factory=lambda: _default_lip_guides(side="lower"))


class Mouth:
    def __init__(
        self,
        *,
        guides: MouthGuides,
        parent: str,
        control_parent: Control | str,
        joint_parent: str,
        face_mid: str,
        jaw: str,
        control_size: float = 1,
        primary_skin: Path,
        secondary_skin: Path,
        tertiary_skin: Path,
        body_skin: Path,
        blendshape_directory: str = "/rig/logic/face/mouth",
        skin_set: str = "main_skin_grp",
    ):
        self.def_group = create_transform("mouth_def", parent=joint_parent)

        self.guides = guides
        duplicated_mouth_surface: str = cmds.duplicate(self.guides.mouth_surface)[0]
        cmds.parent(duplicated_mouth_surface, parent)
        self.mouth_surface = cmds.rename(duplicated_mouth_surface, "mouth_surface")
        cmds.hide(self.mouth_surface)

        duplicated_mouth_surface = cmds.duplicate(self.guides.mouth_surface)[0]
        cmds.parent(duplicated_mouth_surface, parent)
        self.mouth_surface_local = cmds.rename(duplicated_mouth_surface, "mouth_surface_local")

        reference_space = str(control_parent)

        self.mouth_control = create_control(
            "mouth_M",
            transform=guides.mouth,
            parent=control_parent,
            size=control_size * 5,
            control_shape=ControlShape.LINE,
            direction="z",
        )

        self.jaw_blend = create_transform("jaw_M_blend", parent=str(control_parent), transform=jaw)
        cmds.parentConstraint(
            jaw,
            face_mid,
            self.jaw_blend,
            maintainOffset=True,
        )
        matrix_constraint(self.jaw_blend, self.mouth_control.offset)

        self.mouth_slide_reference = create_transform(
            "mouth_M_slide_ref", parent=str(control_parent)
        )
        surface_slide_constraint(
            self.mouth_surface,
            driver_transform=self.mouth_control.offset,
            slider_transform=self.mouth_slide_reference,
        )

        self.mouth_slide = create_transform("mouth_M_slide", parent=self.mouth_control.transform)
        surface_slide_constraint(
            self.mouth_surface,
            driver_transform=self.mouth_control.transform,
            slider_transform=self.mouth_slide,
        )
        self.mouth_local_npo = create_transform(
            "mouth_M_local_npo", transform=self.mouth_control.transform, parent=parent
        )
        self.mouth_local = create_transform("mouth_M_local", parent=self.mouth_local_npo)
        connect_transform(self.mouth_control.transform, self.mouth_local)
        self.mouth_slide_local = create_transform("mouth_M_slide_local", parent=self.mouth_local)
        surface_slide_constraint(
            self.mouth_surface_local,
            driver_transform=self.mouth_local,
            slider_transform=self.mouth_slide_local,
        )

        self.jaw_local_npo = create_transform("jaw_M_local_npo", parent=parent, transform=jaw)
        self.jaw_local = create_transform("jaw_M_local", parent=parent, transform=jaw)
        local_constraint(jaw, self.jaw_local, reference_space=reference_space)
        self.jaw_blend_local = create_transform("jaw_M_blend_local", parent=parent, transform=jaw)
        cmds.parentConstraint(
            self.jaw_local,
            reference_space,
            self.jaw_blend_local,
            maintainOffset=True,
        )

        def _dup_ribbon(name: str) -> str:
            ribbon = cmds.duplicate(guides.lip_ribbon, name=name)[0]
            cmds.parent(ribbon, parent)
            return ribbon

        def _build_layer(mesh: str, *, skin: Path | None = None, below: str | None = None) -> None:
            """Blendshape from the layer below (if any), then skin on top."""
            if below is not None:
                bs = create_blendshape(mesh, parent_directory=blendshape_directory)
                add_blendshape_target_from_geometry(bs, below, below)
            if skin is not None:
                skin_and_apply_weights(skin, mesh)

        self.lip_ribbon_primary = _dup_ribbon("lip_ribbon_primary")
        self.lip_ribbon_secondary = _dup_ribbon("lip_ribbon_secondary")
        self.lip_ribbon_tertiary = _dup_ribbon("lip_ribbon_tertiary")

        self.lip_ribbon_primary_only = _dup_ribbon("lip_ribbon_primary_only")

        self.lip_ribbon_def = _dup_ribbon("lip_ribbon_def")
        self.lip_ribbon_def_uv_pin = setup_uv_pin_node("lip_def_uvPin", self.lip_ribbon_def)

        self.lip_ribbon_primary_attach = _dup_ribbon("lip_ribbon_primary_attach")
        self.lip_ribbon_primary_attach_uv_pin = setup_uv_pin_node(
            "lip_primary_attach_uvPin", self.lip_ribbon_primary_attach
        )

        self.lip_ribbon_secondary_attach = _dup_ribbon("lip_ribbon_secondary_attach")
        self.lip_ribbon_secondary_attach_uv_pin = setup_uv_pin_node(
            "lip_secondary_attach_uvPin", self.lip_ribbon_primary_attach
        )

        self.left_corner = MouthCorner(
            side=Side.LEFT,
            guide=guides.left_corner,
            mouth_surface=self.mouth_surface,
            mouth_surface_local=self.mouth_surface_local,
            control_parent=self.mouth_slide,
            joint_parent=self.def_group,
            parent=parent,
            control_size=control_size,
            def_ribbon=self.lip_ribbon_def,
            def_ribbon_uv_pin=self.lip_ribbon_def_uv_pin,
            mouth_slide_local=self.mouth_slide_local,
        )
        self.right_corner = MouthCorner(
            side=Side.RIGHT,
            guide=guides.right_corner,
            mouth_surface=self.mouth_surface,
            mouth_surface_local=self.mouth_surface_local,
            control_parent=self.mouth_slide,
            joint_parent=self.def_group,
            parent=parent,
            control_size=control_size,
            def_ribbon=self.lip_ribbon_def,
            def_ribbon_uv_pin=self.lip_ribbon_def_uv_pin,
            mouth_slide_local=self.mouth_slide_local,
        )

        self.upper_lip = Lip(
            upper=True,
            guides=guides.upper_lip,
            mouth_surface=self.mouth_surface,
            mouth_surface_local=self.mouth_surface_local,
            parent=parent,
            joint_parent=self.def_group,
            control_parent=self.mouth_slide,
            control_follow=face_mid,
            left_corner=self.left_corner,
            right_corner=self.right_corner,
            mouth_slide=self.mouth_slide,
            mouth_slide_local=self.mouth_slide_local,
            mouth_slide_ref=self.mouth_slide_reference,
            control_size=control_size * 0.5,
            def_ribbon=self.lip_ribbon_def,
            def_ribbon_uv_pin=self.lip_ribbon_def_uv_pin,
            primary_follow_ribbon=self.lip_ribbon_primary_attach,
            primary_follow_ribbon_uv_pin=self.lip_ribbon_primary_attach_uv_pin,
            secondary_follow_ribbon=self.lip_ribbon_secondary_attach,
            secondary_follow_ribbon_uv_pin=self.lip_ribbon_secondary_attach_uv_pin,
            uv_range=(-0.25, 0.25),
        )

        self.lower_lip = Lip(
            upper=False,
            guides=guides.lower_lip,
            mouth_surface=self.mouth_surface,
            mouth_surface_local=self.mouth_surface_local,
            parent=parent,
            joint_parent=self.def_group,
            control_parent=self.mouth_slide,
            control_follow=jaw,
            left_corner=self.left_corner,
            right_corner=self.right_corner,
            mouth_slide=self.mouth_slide,
            mouth_slide_local=self.mouth_slide_local,
            mouth_slide_ref=self.mouth_slide_reference,
            control_size=control_size * 0.5,
            def_ribbon=self.lip_ribbon_def,
            def_ribbon_uv_pin=self.lip_ribbon_def_uv_pin,
            primary_follow_ribbon=self.lip_ribbon_primary_attach,
            primary_follow_ribbon_uv_pin=self.lip_ribbon_primary_attach_uv_pin,
            secondary_follow_ribbon=self.lip_ribbon_secondary_attach,
            secondary_follow_ribbon_uv_pin=self.lip_ribbon_secondary_attach_uv_pin,
            uv_range=(0.75, 0.25),
        )

        _build_layer(self.lip_ribbon_tertiary, skin=tertiary_skin)
        _build_layer(self.lip_ribbon_secondary, skin=secondary_skin, below=self.lip_ribbon_tertiary)
        _build_layer(self.lip_ribbon_primary, skin=primary_skin, below=self.lip_ribbon_secondary)
        _build_layer(self.lip_ribbon_def, below=self.lip_ribbon_primary)

        _build_layer(self.lip_ribbon_primary_only, skin=primary_skin)

        _build_layer(self.lip_ribbon_primary_attach, skin=primary_skin)
        skin_and_apply_weights(body_skin, self.lip_ribbon_primary_attach)

        _build_layer(
            self.lip_ribbon_secondary_attach,
            skin=secondary_skin,
            below=self.lip_ribbon_primary_only,
        )
