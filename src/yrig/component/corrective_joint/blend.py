from maya import cmds
from maya.api.OpenMaya import MMatrix

from yrig.joint import create_joint
from yrig.maya_api.attribute import DoubleAttribute, IntegerAttribute
from yrig.maya_api.node import (
    DecomposeMatrixNode,
    QuatNormalizeNode,
    QuatSlerpNode,
    QuatToEulerNode,
    RemapValueNode,
)
from yrig.name import get_short_name
from yrig.transform import create_transform, matrix_constraint
from yrig.transform.matrix import get_world_matrix, localize_world_matrix, multiply_matrices
from yrig.transform.quat import drive_rotation_with_quat


class BlendCorrective:
    """Build one guide-positioned deform joint with a rest-relative rotation correction."""

    def __init__(
        self,
        guide: str,
        driver: str,
        joint_parent: str,
        parent: str = "rig",
        deform_parent: str | None = None,
        part: str = "corrective",
        side: str = "M",
        blend: float = 0.5,
        driver_axis: str | None = None,
        driver_rot_range: tuple[float, float] = (-90.0, 90.0),
        mult_range: tuple[float, float] = (0.0, 1.0),
        translate_blend: float = 0.0,
        pop_amount: float | None = None,
        pop_axis: str = "Y",
    ) -> None:
        self.guide = guide
        self.driver = driver
        self.joint_parent = joint_parent
        self.parent = parent
        self.deform_parent = deform_parent or joint_parent
        self.part = part
        self.side = side
        self.blend = blend
        self.driver_axis = driver_axis.upper() if driver_axis else None
        self.driver_rot_range = driver_rot_range
        self.mult_range = mult_range
        self.translate_blend = translate_blend
        self.pop_amount = pop_amount
        self.pop_axis = pop_axis.upper()

        self.descriptor = get_short_name(guide).removesuffix("_guide")
        self.name = f"{self.part}_{self.side}_{self.descriptor}"
        self._validate()

    def _validate(self) -> None:
        for label, node in (
            ("guide", self.guide),
            ("driver", self.driver),
            ("joint_parent", self.joint_parent),
            ("parent", self.parent),
            ("deform_parent", self.deform_parent),
        ):
            if not cmds.objExists(node):
                raise ValueError(f"{label} does not exist: {node}")

        if self.driver_axis not in (None, "X", "Y", "Z"):
            raise ValueError("driver_axis must be one of 'X', 'Y', 'Z', or None")
        if self.pop_axis not in ("X", "Y", "Z"):
            raise ValueError("pop_axis must be one of 'X', 'Y', or 'Z'")
        if self.pop_amount is not None and self.driver_axis is None:
            raise ValueError("pop_amount requires driver_axis")
        if self.driver_rot_range[0] == self.driver_rot_range[1]:
            raise ValueError("driver_rot_range endpoints must differ")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be between 0 and 1")
        if not 0.0 <= self.translate_blend <= 1.0:
            raise ValueError("translate_blend must be between 0 and 1")

    def setup_structure(self) -> None:
        self.main_grp = create_transform(
            name=f"{self.name}_grp",
            parent=self.parent,
        )
        self.component_grp = create_transform(
            name=f"{self.name}_component_grp",
            parent=self.main_grp,
        )
        cmds.hide(self.component_grp)

    def _build_relative_delta(self) -> DecomposeMatrixNode:
        relative_matrix = localize_world_matrix(self.driver, self.joint_parent)
        rest_relative_matrix: MMatrix = (
            get_world_matrix(self.driver) * get_world_matrix(self.joint_parent).inverse()
        )
        delta_matrix = multiply_matrices(
            f"{self.name}_relative_delta",
            (relative_matrix.matrix_sum, rest_relative_matrix.inverse()),
        )
        delta = DecomposeMatrixNode.create(f"{self.name}_delta_decompose")
        delta.input_matrix.connect_from(delta_matrix.matrix_sum)
        return delta

    def _build_signed_twist(self, delta: DecomposeMatrixNode) -> str:
        driver_axis = self.driver_axis
        if driver_axis is None:
            raise RuntimeError("A driver axis is required to build signed twist")

        twist = QuatNormalizeNode.create(f"{self.name}_twist_normalize")
        twist.input_quat.w.connect_from(delta.output_quat.w)
        for component in "xyz":
            destination = getattr(twist.input_quat, component)
            if component == driver_axis.lower():
                destination.connect_from(getattr(delta.output_quat, component))
            else:
                destination.set(0.0)

        twist_euler = QuatToEulerNode.create(f"{self.name}_twist_euler")
        twist_euler.input_quat.connect_from(twist.output_quat)
        twist_euler.input_rotate_order.set(0)
        angle = getattr(twist_euler.output_rotate, driver_axis.lower())
        return str(angle)

    def _build_weight(self, signed_twist: str | None) -> str | None:
        if signed_twist is None:
            return None

        activation = RemapValueNode.create(f"{self.name}_activation_remap")
        activation.input_min.set(self.driver_rot_range[0])
        activation.input_max.set(self.driver_rot_range[1])
        activation.output_min.set(self.mult_range[0])
        activation.output_max.set(self.mult_range[1])
        activation.input_value.connect_from(signed_twist)

        blend_mult = cmds.createNode("multDoubleLinear", name=f"{self.name}_blend_weight")
        activation.output.connect_to(f"{blend_mult}.input1")
        DoubleAttribute(f"{blend_mult}.input2").set(self.blend)
        return f"{blend_mult}.output"

    def _build_pop(self, signed_twist: str | None) -> str | None:
        if self.pop_amount is None:
            return None
        if signed_twist is None:
            raise RuntimeError("A pop requires a signed twist input")

        pop_remap = RemapValueNode.create(f"{self.name}_pop_remap")
        pop_remap.input_min.set(self.driver_rot_range[0])
        pop_remap.input_max.set(self.driver_rot_range[1])
        pop_remap.output_min.set(0.0)
        pop_remap.output_max.set(self.pop_amount)
        pop_remap.input_value.connect_from(signed_twist)

        pop_transform = create_transform(
            name=f"{self.name}_pop",
            parent=self.switch_joint,
        )
        pop_remap.output.connect_to(f"{pop_transform}.translate{self.pop_axis}")
        return pop_transform

    def _connect_translation(
        self,
        delta: DecomposeMatrixNode,
        weight: str | None,
    ) -> None:
        if not self.translate_blend:
            return

        translate_mult = cmds.createNode("multiplyDivide", name=f"{self.name}_translate_mult")
        IntegerAttribute(f"{translate_mult}.operation").set(1)
        delta.output_translate.connect_to(f"{translate_mult}.input1")

        if weight is None:
            value = self.blend * self.translate_blend
            for axis in "XYZ":
                DoubleAttribute(f"{translate_mult}.input2{axis}").set(value)
        else:
            weight_mult = cmds.createNode("multDoubleLinear", name=f"{self.name}_translate_weight")
            cmds.connectAttr(weight, f"{weight_mult}.input1")
            DoubleAttribute(f"{weight_mult}.input2").set(self.translate_blend)
            for axis in "XYZ":
                cmds.connectAttr(f"{weight_mult}.output", f"{translate_mult}.input2{axis}")

        cmds.connectAttr(f"{translate_mult}.output", f"{self.switch_blend}.translate")

    def build(self) -> str:
        """Create the corrective hierarchy and return its registered deform joint."""
        self.setup_structure()

        self.switch_parent = create_transform(
            name=f"{self.name}_switch_parent",
            parent=self.component_grp,
            transform=self.joint_parent,
        )
        matrix_constraint(
            self.joint_parent,
            self.switch_parent,
            keep_offset=False,
            scale=False,
            shear=False,
        )

        self.switch_blend = create_transform(
            name=f"{self.name}_switch_blend",
            parent=self.switch_parent,
        )
        self.switch_joint = create_transform(
            name=f"{self.name}_switch",
            parent=self.switch_blend,
            transform=self.guide,
        )

        delta = self._build_relative_delta()
        signed_twist = self._build_signed_twist(delta) if self.driver_axis is not None else None
        weight = self._build_weight(signed_twist)

        rotation_blend = QuatSlerpNode.create(f"{self.name}_rotation_slerp")
        rotation_blend.input1_quat.x.set(0.0)
        rotation_blend.input1_quat.y.set(0.0)
        rotation_blend.input1_quat.z.set(0.0)
        rotation_blend.input1_quat.w.set(1.0)
        rotation_blend.input2_quat.connect_from(delta.output_quat)
        if weight is None:
            rotation_blend.input_t.set(self.blend)
        else:
            rotation_blend.input_t.connect_from(weight)
        IntegerAttribute(f"{rotation_blend.name}.angleInterpolation").set(1)
        drive_rotation_with_quat(self.switch_blend, rotation_blend.output_quat)

        self._connect_translation(delta, weight)
        output_transform = self._build_pop(signed_twist) or self.switch_joint

        self.blend_joint = create_joint(
            name=f"{self.name}_def",
            transform=self.guide,
            parent=self.deform_parent,
            connect=False,
        )
        matrix_constraint(
            output_transform,
            self.blend_joint,
            keep_offset=False,
            use_joint_orient=True,
            scale=False,
            shear=False,
        )
        return self.blend_joint


__all__ = ["BlendCorrective"]
