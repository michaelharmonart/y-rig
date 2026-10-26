import math

from maya import cmds
from maya.api.OpenMaya import (
    MEulerRotation,
    MFnNurbsCurve,
    MMatrix,
    MPoint,
    MSelectionList,
    MSpace,
    MTransformationMatrix,
    MVector,
)

from yrig.control import Control, ControlShape, create_control
from yrig.joint import create_joint
from yrig.maya_api.node import SumNode
from yrig.skin.split.tag import tag_for_weight_split
from yrig.transform import create_transform

from .ik import create_ik_single_chain
from .mirror import mirror_matrix


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
        setup_parent: str = "logic",
        setup_group: str = "eyes_setup",
        orient_sub_expression_controls: bool = False,
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
        self.setup_parent = setup_parent
        self.setup_group = setup_group

        self.blink_guides = {}
        self.blink_curves = {}
        self.blink_blendshapes = {}
        self.orient_sub_expression_controls = orient_sub_expression_controls

    # -------------------
    # Helper Functions
    # -------------------

    def build_eyelid_collision(
        self,
        push: float = 0.5,
    ) -> None:
        """Build zero-gap collision between corresponding upper/lower IK pins.

        Requires:
            self.eyelid_ik["upper"]
            self.eyelid_ik["lower"]

        Each entry must contain:
            "pin": curve-pinned transform
            "collision": collision offset parented under the pin
        """

        # ---------------------------------------------------------
        # Collision gap attribute
        # ---------------------------------------------------------
        if not cmds.attributeQuery(
            "collision",
            node=self.main_ctrl,
            exists=True,
        ):
            cmds.addAttr(
                self.main_ctrl,
                longName="collision",
                attributeType="double",
                minValue=0.0,
                maxValue=1.0,
                defaultValue=1.0,
                keyable=True,
            )

        if not cmds.attributeQuery(
            "collisionGap",
            node=self.main_ctrl,
            exists=True,
        ):
            cmds.addAttr(
                self.main_ctrl,
                longName="collisionGap",
                attributeType="double",
                defaultValue=0.07,
                keyable=True,
            )

        upper_items = self.eyelid_ik["upper"]
        lower_items = self.eyelid_ik["lower"]

        if len(upper_items) != len(lower_items):
            raise RuntimeError("Upper and lower eyelids must have matching IK pin counts.")

        for index, (upper, lower) in enumerate(zip(upper_items, lower_items, strict=True)):
            self.create_eyelid_collision_pair(
                upper_pin=upper["pin"],
                lower_pin=lower["pin"],
                upper_collision=upper["collision"],
                lower_collision=lower["collision"],
                name=f"eyelid_collision_{index:02d}_{self.side}",
                push=push,
            )

    def get_expression_mirrored_matrix(
        self,
        matrix: MMatrix,
    ) -> MMatrix:
        """Apply left-to-right orientation behavior to expression matrices.

        The supplied matrix is already at the correct world position.
        Right-side orientation is constructed as a mirrored left-side
        orientation without moving the control.
        """

        if self.side != "R":
            return matrix

        # Reflect the right-side matrix into hypothetical left space.
        left_matrix = mirror_matrix(
            matrix,
            axis="x",
            behavior=False,
        )

        # Mirror the left-style transform back with proper handedness.
        return mirror_matrix(
            left_matrix,
            axis="x",
            behavior=True,
        )

    def create_eyelid_collision_pair(
        self,
        upper_pin: str,
        lower_pin: str,
        upper_collision: str,
        lower_collision: str,
        name: str,
        push: float = 0.5,
    ) -> None:
        """Blend upper/lower IK drivers to a shared midpoint when crossing.

        Assumes upper_collision and lower_collision are independent
        transforms, not children of their respective pins.
        """

        push = max(0.0, min(1.0, push))

        # ---------------------------------------------------------
        # 1. Create shared midpoint
        # ---------------------------------------------------------

        midpoint = cmds.createNode(
            "transform",
            name=f"{name}_midpoint",
            parent=self.collision_grp,
        )

        midpoint_blend = cmds.createNode(
            "blendMatrix",
            name=f"{name}_midpoint_BM",
        )

        cmds.connectAttr(
            f"{upper_pin}.worldMatrix[0]",
            f"{midpoint_blend}.inputMatrix",
        )

        cmds.connectAttr(
            f"{lower_pin}.worldMatrix[0]",
            f"{midpoint_blend}.target[0].targetMatrix",
        )

        cmds.setAttr(
            f"{midpoint_blend}.target[0].weight",
            push,  # type:ignore
        )

        # Match the midpoint's parent space.
        midpoint_mm = cmds.createNode(
            "multMatrix",
            name=f"{name}_midpoint_MM",
        )

        cmds.connectAttr(
            f"{midpoint_blend}.outputMatrix",
            f"{midpoint_mm}.matrixIn[0]",
        )

        cmds.connectAttr(
            f"{self.collision_grp}.worldInverseMatrix[0]",
            f"{midpoint_mm}.matrixIn[1]",
        )

        midpoint_dcm = cmds.createNode(
            "decomposeMatrix",
            name=f"{name}_midpoint_DCM",
        )

        cmds.connectAttr(
            f"{midpoint_mm}.matrixSum",
            f"{midpoint_dcm}.inputMatrix",
        )

        cmds.connectAttr(
            f"{midpoint_dcm}.outputTranslate",
            f"{midpoint}.translate",
        )

        cmds.connectAttr(
            f"{midpoint_dcm}.outputRotate",
            f"{midpoint}.rotate",
        )

        # ---------------------------------------------------------
        # 2. Determine when upper/lower pins cross
        # ---------------------------------------------------------

        def get_local_y(pin: str, suffix: str) -> str:
            """Get pin Y relative to the isolated eye setup."""

            mm = cmds.createNode(
                "multMatrix",
                name=f"{name}_{suffix}_compare_MM",
            )

            dcm = cmds.createNode(
                "decomposeMatrix",
                name=f"{name}_{suffix}_compare_DCM",
            )

            cmds.connectAttr(
                f"{pin}.worldMatrix[0]",
                f"{mm}.matrixIn[0]",
            )

            cmds.connectAttr(
                f"{self.eyes_setup_grp}.worldInverseMatrix[0]",
                f"{mm}.matrixIn[1]",
            )

            cmds.connectAttr(
                f"{mm}.matrixSum",
                f"{dcm}.inputMatrix",
            )

            return f"{dcm}.outputTranslateY"

        upper_y = get_local_y(upper_pin, "upper")
        lower_y = get_local_y(lower_pin, "lower")

        difference = cmds.createNode(
            "plusMinusAverage",
            name=f"{name}_difference_PMA",
        )

        cmds.setAttr(f"{difference}.operation", 2)  # type:ignore

        cmds.connectAttr(
            lower_y,
            f"{difference}.input1D[0]",
        )

        cmds.connectAttr(
            upper_y,
            f"{difference}.input1D[1]",
        )

        # ---------------------------------------------------------
        # Smooth collision activation
        # ---------------------------------------------------------

        soft_distance = self.control_size * 0.1

        collision_remap = cmds.createNode(
            "remapValue",
            name=f"{name}_collision_RMV",
        )

        cmds.setAttr(
            f"{collision_remap}.inputMin",
            -soft_distance,  # type:ignore
        )

        cmds.setAttr(
            f"{collision_remap}.inputMax",
            0.0,  # type:ignore
        )

        cmds.setAttr(
            f"{collision_remap}.outputMin",
            0.0,  # type:ignore
        )

        cmds.setAttr(
            f"{collision_remap}.outputMax",
            1.0,  # type:ignore
        )

        cmds.connectAttr(
            f"{difference}.output1D",
            f"{collision_remap}.inputValue",
        )

        # Smooth interpolation
        cmds.setAttr(
            f"{collision_remap}.value[0].value_Interp",
            3,  # type:ignore
        )

        cmds.setAttr(
            f"{collision_remap}.value[1].value_Interp",
            3,  # type:ignore
        )

        # ---------------------------------------------------------
        # Animator collision strength
        # ---------------------------------------------------------

        collision_strength = cmds.createNode(
            "multiplyDivide",
            name=f"{name}_collision_strength_MD",
        )

        cmds.connectAttr(
            f"{collision_remap}.outValue",
            f"{collision_strength}.input1X",
        )

        cmds.connectAttr(
            f"{self.main_ctrl}.collision",
            f"{collision_strength}.input2X",
        )

        collision_weight = f"{collision_strength}.outputX"

        # ---------------------------------------------------------
        # 3. Blend each collision driver to the midpoint
        # ---------------------------------------------------------

        for pin, collision, suffix in (
            (upper_pin, upper_collision, "upper"),
            (lower_pin, lower_collision, "lower"),
        ):
            blend = cmds.createNode(
                "blendMatrix",
                name=f"{name}_{suffix}_collision_BM",
            )

            cmds.connectAttr(
                f"{pin}.worldMatrix[0]",
                f"{blend}.inputMatrix",
            )

            cmds.connectAttr(
                f"{midpoint}.worldMatrix[0]",
                f"{blend}.target[0].targetMatrix",
            )

            cmds.connectAttr(
                collision_weight,
                f"{blend}.target[0].weight",
            )

            parent = cmds.listRelatives(
                collision,
                parent=True,
                fullPath=True,
            )

            mm = cmds.createNode(
                "multMatrix",
                name=f"{name}_{suffix}_local_MM",
            )

            cmds.connectAttr(
                f"{blend}.outputMatrix",
                f"{mm}.matrixIn[0]",
            )

            if parent:
                cmds.connectAttr(
                    f"{parent[0]}.worldInverseMatrix[0]",
                    f"{mm}.matrixIn[1]",
                )

            dcm = cmds.createNode(
                "decomposeMatrix",
                name=f"{name}_{suffix}_local_DCM",
            )

            cmds.connectAttr(
                f"{mm}.matrixSum",
                f"{dcm}.inputMatrix",
            )

            # ---------------------------------------------------------
            # Collision gap offset
            # ---------------------------------------------------------

            gap_mult = cmds.createNode(
                "multiplyDivide",
                name=f"{name}_{suffix}_gap_MD",
            )

            # Upper moves downward, lower moves upward.
            direction = -0.5 if suffix == "upper" else 0.5

            cmds.connectAttr(
                f"{self.main_ctrl}.collisionGap",
                f"{gap_mult}.input1X",
            )

            cmds.setAttr(
                f"{gap_mult}.input2X",
                direction,  # type:ignore
            )

            # Only apply the gap while collision is active.
            gap_active = cmds.createNode(
                "multiplyDivide",
                name=f"{name}_{suffix}_gap_active_MD",
            )

            cmds.connectAttr(
                f"{gap_mult}.outputX",
                f"{gap_active}.input1X",
            )

            cmds.connectAttr(
                collision_weight,
                f"{gap_active}.input2X",
            )

            # Add the offset to the existing collision translation.
            gap_sum = cmds.createNode(
                "plusMinusAverage",
                name=f"{name}_{suffix}_gap_PMA",
            )

            cmds.connectAttr(
                f"{dcm}.outputTranslate",
                f"{gap_sum}.input3D[0]",
            )

            cmds.connectAttr(
                f"{gap_active}.outputX",
                f"{gap_sum}.input3D[1].input3Dy",
            )

            cmds.connectAttr(
                f"{gap_sum}.output3D",
                f"{collision}.translate",
                force=True,
            )

            cmds.connectAttr(
                f"{dcm}.outputRotate",
                f"{collision}.rotate",
                force=True,
            )

    def setup_sub_eyelid_visibility(self) -> None:
        """Control sub-eyelid visibility from the main eye control."""

        main_ctrl = self.main_ctrl

        # Main visibility attributes

        cmds.addAttr(
            main_ctrl,
            longName="sub_eyelid_vis",
            attributeType="bool",
            defaultValue=False,
            keyable=True,
        )

        visibility_attr = f"{main_ctrl}.sub_eyelid_vis"

        # Proxy attribute on blink controls
        for control in self.blink_controls.values():
            cmds.addAttr(
                control.transform,
                longName="sub_eyelid",
                proxy=visibility_attr,
            )

        # Sub-eyelid control visibility
        for vertical in ("upper", "lower"):
            for control in self.sub_eyelid_controls[vertical]:
                cmds.connectAttr(
                    visibility_attr,
                    f"{control.transform}.visibility",
                    force=True,
                )

                cmds.addAttr(
                    control.transform,
                    longName="sub_eyelid",
                    proxy=visibility_attr,
                )

    def connect_major_controls(self) -> None:
        """Connect real major controls to fake drivers.

        Mid controls receive additive blink translation X and rotation Z.
        Other channels connect directly.
        """

        for name, control in self.expression_major_controls.items():
            fake_driver = self.expression_major_drivers[name]

            # ---------------------------------------------------------
            # Direct connections for all channels
            # ---------------------------------------------------------

            for axis in "XYZ":
                cmds.connectAttr(
                    f"{control.transform}.translate{axis}",
                    f"{fake_driver}.translate{axis}",
                    force=True,
                )

                cmds.connectAttr(
                    f"{control.transform}.rotate{axis}",
                    f"{fake_driver}.rotate{axis}",
                    force=True,
                )

            # ---------------------------------------------------------
            # Additive behavior for upper/lower midpoint
            # ---------------------------------------------------------

            if name not in ("upper_mid", "lower_mid"):
                continue

            vertical = name.split("_")[0]
            blink_control = self.blink_controls[vertical].transform

            for attribute in ("translateX", "rotateZ"):
                sum_node = SumNode.create(
                    name=f"eyelid_{name}_{attribute}_{self.side}_SUM",
                )

                sum_node.input[0].connect_from(f"{control.transform}.{attribute}")
                sum_node.input[1].connect_from(f"{blink_control}.{attribute}")

                destination = f"{fake_driver}.{attribute}"

                # Disconnect any existing incoming connection.
                existing = (
                    cmds.listConnections(
                        destination,
                        source=True,
                        destination=False,
                        plugs=True,
                    )
                    or []
                )

                for source in existing:
                    cmds.disconnectAttr(source, destination)

                sum_node.output.connect_to(destination)

    def smooth_macro_curve_weights(self) -> None:
        """Smooth upper and lower macro curve skin weights."""

        for vertical in ("upper", "lower"):
            self.smooth_curve_weights(
                curve=self.expression_rest_curves[vertical],
                iterations=5,
                strength=0.5,
                preserve_ends=True,
            )

    def smooth_curve_weights(
        self,
        curve: str,
        iterations: int = 5,
        strength: float = 0.5,
        preserve_ends: bool = True,
    ) -> None:
        """Smooth skin weights along a NURBS curve."""

        strength = max(0.0, min(1.0, strength))

        history = cmds.listHistory(curve) or []
        skins = cmds.ls(history, type="skinCluster")  # type:ignore

        if not skins:
            raise ValueError(f"No skinCluster found on {curve}")

        skin = skins[0]

        influences = cmds.skinCluster(
            skin,
            query=True,
            influence=True,
        )

        cvs = cmds.ls(
            f"{curve}.cv[*]",
            flatten=True,
        )

        if not cvs:
            raise ValueError(f"No CVs found on {curve}")

        for _ in range(iterations):
            weights = [
                [
                    cmds.skinPercent(
                        skin,
                        cv,
                        query=True,
                        transform=joint,
                    )
                    for joint in influences  # type:ignore
                ]
                for cv in cvs
            ]

            for i, cv in enumerate(cvs):
                if preserve_ends and i in (0, len(cvs) - 1):
                    continue

                neighbors = [weights[j] for j in (i - 1, i + 1) if 0 <= j < len(cvs)]

                if not neighbors:
                    continue

                new_weights = []

                for j, weight in enumerate(weights[i]):
                    average = sum(neighbor[j] for neighbor in neighbors) / len(neighbors)  # type:ignore

                    smoothed = weight * (1.0 - strength) + average * strength  # type:ignore

                    new_weights.append(smoothed)

                cmds.skinPercent(
                    skin,
                    cv,
                    transformValue=list(zip(influences, new_weights, strict=True)),  # type:ignore
                    normalize=True,
                )

    def build_macro_joints(self) -> None:
        """Create a driven joint for every upper/lower macro control."""

        self.macro_joints = {
            "upper": [],
            "lower": [],
        }

        for vertical in ("upper", "lower"):
            controls = self.expression_macro_controls[vertical]
            fake_drivers = self.expression_macro_drivers[vertical]

            for index, (control, fake_driver) in enumerate(
                zip(controls, fake_drivers, strict=True)
            ):
                # ---------------------------------------------
                # Real control drives fake control
                # ---------------------------------------------

                cmds.connectAttr(
                    f"{control.transform}.translate",
                    f"{fake_driver}.translate",
                    force=True,
                )

                cmds.connectAttr(
                    f"{control.transform}.rotate",
                    f"{fake_driver}.rotate",
                    force=True,
                )

                # ---------------------------------------------
                # Fake control drives macro joint
                # ---------------------------------------------

                joint = create_joint(
                    name=f"eyelid_{vertical}_macro_{index:02d}_{self.side}",
                    parent=self.expression_driver_grp,
                    transform=fake_driver,
                    connect=True,
                )

                self.macro_joints[vertical].append(joint)

    def skin_macro_curves(self) -> None:
        """Skin each eyelid curve to its five macro joints."""

        self.macro_skin_clusters = {}

        for vertical in ("upper", "lower"):
            curve = self.expression_rest_curves[vertical]
            joints = self.macro_joints[vertical]

            skin = cmds.skinCluster(
                *joints,
                curve,
                toSelectedBones=True,
                bindMethod=0,
                skinMethod=0,
                normalizeWeights=1,
                maximumInfluences=5,
                name=f"{vertical}_macro_{self.side}_skin",
            )[0]  # type:ignore

            self.macro_skin_clusters[vertical] = skin

    def create_expression_control_pair(
        self,
        name: str,
        matrix: MMatrix,
        control_parent: str,
        fake_parent: str,
        size: float,
        mirror_offset: bool = False,
    ) -> tuple[Control, str]:
        """Create matching real and fake expression controls.

        Mirroring is applied to both offset hierarchies.

        Only mirror when requested, avoiding additional mirroring
        for controls placed using explicit guides.
        """

        # ---------------------------------------------------------
        # Real control
        # ---------------------------------------------------------

        control = create_control(
            name=name,
            parent=control_parent,
            transform=matrix,
            size=size,
            control_shape="hexagon",
            direction="z",
        )

        # ---------------------------------------------------------
        # Fake offset
        # ---------------------------------------------------------

        fake_offset = create_transform(
            name=f"{name}_fake_offset",
            parent=fake_parent,
            transform=control.transform,
        )

        # ---------------------------------------------------------
        # Mirror both offsets
        # ---------------------------------------------------------

        if mirror_offset:
            cmds.setAttr(
                f"{control.offset}.scaleX",
                -1.0,  # type:ignore
            )

            cmds.setAttr(
                f"{fake_offset}.scaleX",
                -1.0,  # type:ignore
            )

        # ---------------------------------------------------------
        # Fake driver
        # ---------------------------------------------------------

        fake_driver = create_transform(
            name=f"{name}_fake_driver",
            parent=fake_offset,
            transform=control.transform,
        )

        return control, fake_driver

    def build_expression_controls(self) -> None:
        """Build major and macro eyelid expression controls.

        Optional placement guides are discovered by Maya node name:

            {vertical}_macro_{index:02d}_{side}

        Corners use guide indices 00 and 05.
        Midpoint major controls share guide 02 with macro index 2.

        If a guide is missing, placement falls back to the rest curve.

        Right-side curve-generated controls receive scaleX = -1
        on their offset groups. Explicit guides retain their
        existing orientation.

        Shared corners are parented under the main eye control.
        """

        # ---------------------------------------------------------
        # Setup groups
        # ---------------------------------------------------------

        self.expression_control_grps = {}

        for vertical in ("upper", "lower"):
            self.expression_control_grps[vertical] = create_transform(
                name=f"eyelid_{vertical}_expression_controls_{self.side}_grp",
                parent=self.blink_controls[vertical].transform,
            )

        self.expression_driver_grp = create_transform(
            name=f"expression_drivers_{self.side}_grp",
            parent=self.eyes_setup_grp,
        )

        self.setup_expression_follow()

        self.expression_major_controls = {}
        self.expression_major_drivers = {}

        self.expression_macro_controls = {
            "upper": [],
            "lower": [],
        }

        self.expression_macro_drivers = {
            "upper": [],
            "lower": [],
        }

        upper_curve = self.guides["eyelid_upper_rest"]
        lower_curve = self.guides["eyelid_lower_rest"]

        curves = {
            "upper": upper_curve,
            "lower": lower_curve,
        }

        macro_parameters = (0.0, 0.25, 0.5, 0.75, 1.0)

        # ---------------------------------------------------------
        # Guide / curve placement
        # ---------------------------------------------------------

        def get_placement_matrix(
            vertical: str,
            guide_index: int,
            parameter: float,
        ) -> tuple[MMatrix, bool]:
            """Return placement matrix and whether an explicit guide was used.

            Explicit guides provide position and orientation.
            Curve-generated placements use the curve orientation setting.
            """

            guide_name = f"{vertical}_macro_{guide_index:02d}_{self.side}"

            matches = (
                cmds.ls(
                    guide_name,
                    long=True,
                    type="transform",
                )
                or []
            )

            if len(matches) > 1:
                raise RuntimeError(f"Multiple expression guides found: {guide_name}")

            if matches:
                matrix = MMatrix(
                    cmds.xform(
                        matches[0],
                        query=True,
                        worldSpace=True,
                        matrix=True,
                    )
                )

                return matrix, True

            matrix = self.get_curve_point_matrix(
                curve=curves[vertical],
                parameter=parameter,
                tangent_orient=self.orient_sub_expression_controls,
            )

            return matrix, False

        # ---------------------------------------------------------
        # Cache macro placement matrices and mirror flags
        # ---------------------------------------------------------

        macro_matrices = {
            "upper": {},
            "lower": {},
        }

        macro_mirror = {
            "upper": {},
            "lower": {},
        }

        for vertical in ("upper", "lower"):
            for index, parameter in enumerate(macro_parameters):
                # Macro index 4 uses guide 05.
                guide_index = 5 if index == 4 else index

                # Shared corners use upper eyelid placement.
                if index in (0, 4):
                    placement_vertical = "upper"
                else:
                    placement_vertical = vertical

                matrix, has_guide = get_placement_matrix(
                    vertical=placement_vertical,
                    guide_index=guide_index,
                    parameter=parameter,
                )

                macro_matrices[vertical][index] = matrix

                # Mirror only right-side curve-generated placements.
                macro_mirror[vertical][index] = self.side == "R" and not has_guide

        # ---------------------------------------------------------
        # Shared corner controls
        # ---------------------------------------------------------

        for corner, index in (
            ("inner_corner", 0),
            ("outer_corner", 4),
        ):
            matrix = macro_matrices["upper"][index]

            control, driver = self.create_expression_control_pair(
                name=f"eyelid_{corner}_{self.side}",
                matrix=matrix,
                control_parent=self.main_ctrl,
                fake_parent=self.expression_driver_grp,
                size=self.control_size * 0.2,
                mirror_offset=macro_mirror["upper"][index],
            )

            self.expression_major_controls[corner] = control
            self.expression_major_drivers[corner] = driver

        # ---------------------------------------------------------
        # Upper/lower midpoint controls
        # ---------------------------------------------------------

        for vertical in ("upper", "lower"):
            matrix = macro_matrices[vertical][2]

            control, driver = self.create_expression_control_pair(
                name=f"eyelid_{vertical}_mid_{self.side}",
                matrix=matrix,
                control_parent=self.expression_control_grps[vertical],
                fake_parent=self.expression_follow_grp,
                size=self.control_size * 0.25,
                mirror_offset=macro_mirror[vertical][2],
            )

            self.expression_major_controls[f"{vertical}_mid"] = control
            self.expression_major_drivers[f"{vertical}_mid"] = driver

        # ---------------------------------------------------------
        # Macro controls
        # ---------------------------------------------------------

        for vertical in ("upper", "lower"):
            for index in range(5):
                if index == 0:
                    parent_key = "inner_corner"
                elif index == 4:
                    parent_key = "outer_corner"
                else:
                    parent_key = f"{vertical}_mid"

                major_control = self.expression_major_controls[parent_key]
                major_driver = self.expression_major_drivers[parent_key]

                matrix = macro_matrices[vertical][index]

                control, driver = self.create_expression_control_pair(
                    name=f"eyelid_{vertical}_macro_{index:02d}_{self.side}",
                    matrix=matrix,
                    control_parent=major_control.transform,
                    fake_parent=major_driver,
                    size=self.control_size * 0.12,
                    mirror_offset=macro_mirror[vertical][index],
                )

                self.expression_macro_controls[vertical].append(control)
                self.expression_macro_drivers[vertical].append(driver)

    def create_sub_control_ik_driver(
        self,
        control: Control,
        end_joint: str,
    ) -> str:
        """Drive a real SDK using an IK-constrained proxy SDK.

        The proxy hierarchy lives in the static eye setup.

        The proxy offset matches the real SDK's rest pose.
        The proxy SDK follows the IK with maintainOffset=True.
        Its local TR channels drive the real SDK directly.
        """

        transform = control.transform

        # ---------------------------------------------------------
        # Create the real SDK
        # ---------------------------------------------------------

        sdk = cmds.createNode(
            "transform",
            name=f"{transform}_sdk",
            parent=control.offset,
        )

        cmds.parent(
            transform,
            sdk,
            relative=True,
        )

        # ---------------------------------------------------------
        # Create proxy offset
        # ---------------------------------------------------------

        proxy_offset = create_transform(
            name=f"{transform}_ik_offset",
            parent=self.ik_driver_grp,
            transform=sdk,
        )

        # ---------------------------------------------------------
        # Create proxy SDK
        #
        # Same world transform as proxy offset.
        # Local translate/rotate should be zero.
        # ---------------------------------------------------------

        proxy_sdk = create_transform(
            name=f"{transform}_ik_sdk",
            parent=proxy_offset,
            transform=sdk,
        )

        # ---------------------------------------------------------
        # Constrain proxy SDK to IK end joint
        # ---------------------------------------------------------

        cmds.parentConstraint(
            end_joint,
            proxy_sdk,
            maintainOffset=True,
        )

        # ---------------------------------------------------------
        # Drive real SDK from proxy SDK
        # ---------------------------------------------------------

        cmds.connectAttr(
            f"{proxy_sdk}.translate",
            f"{sdk}.translate",
            force=True,
        )

        cmds.connectAttr(
            f"{proxy_sdk}.rotate",
            f"{sdk}.rotate",
            force=True,
        )

        # ---------------------------------------------------------
        # Store references
        # ---------------------------------------------------------

        control.sdk = sdk  # type: ignore

        return sdk

    def connect_sub_controls_to_ik(self) -> None:
        """Connect sub-controls to their IK proxy drivers."""

        self.ik_driver_grp = create_transform(
            name=f"eyelid_ik_drivers_{self.side}_grp",
            parent=self.eyes_setup_grp,
        )

        self.sub_control_sdks = {
            "upper": [],
            "lower": [],
        }

        for vertical in ("upper", "lower"):
            controls = self.sub_eyelid_controls[vertical]
            chains = self.eyelid_ik[vertical]

            if len(controls) != len(chains):
                raise ValueError(
                    f"{vertical} eyelid control/IK count mismatch: "
                    f"{len(controls)} controls, "
                    f"{len(chains)} IK chains."
                )

            for control, chain in zip(
                controls,
                chains,
                strict=True,
            ):
                sdk = self.create_sub_control_ik_driver(
                    control=control,
                    end_joint=chain["end_joint"],
                )

                self.sub_control_sdks[vertical].append(sdk)

    def orient_ik_end_to_curve(
        self,
        name: str,
        curve: str,
        end_joint: str,
        parameter: float,
        parent: str,
    ) -> str:
        """
        Orient the IK end joint from the curve tangent.

        Translation remains controlled by the IK solver.
        """

        driver = create_transform(
            name=f"{name}_orient_driver",
            parent=parent,
        )

        # Initialize at the joint's world orientation.
        cmds.xform(
            driver,
            worldSpace=True,
            matrix=cmds.xform(
                end_joint,
                query=True,
                worldSpace=True,
                matrix=True,
            ),  # type:ignore
        )

        cmds.pointOnCurve(
            curve,
            parameter=parameter,
            turnOnPercentage=True,
            position=True,
        )

        cmds.tangentConstraint(
            curve,
            driver,
            aimVector=(1, 0, 0),
            upVector=(0, 1, 0),
            worldUpType="vector",
            worldUpVector=(0, 1, 0),
        )

        cmds.orientConstraint(
            driver,
            end_joint,
            maintainOffset=True,
        )

        return driver

    def connect_ik_stretch_compress(
        self,
        name: str,
        root_joint: str,
        end_joint: str,
        ik_handle: str,
        control: str,
        rest_length: float,
    ) -> None:
        """
        Add optional stretch/compression to a single-chain IK.

        Stretch and compression are controlled independently.

        Args:
            name: Node name prefix.
            root_joint: Start joint.
            end_joint: End joint.
            ik_handle: IK handle transform.
            control: Blink control transform.
            rest_length: Original root-to-end distance.
        """

        if rest_length <= 0.000001:
            raise ValueError(f"Invalid IK rest length for {name}: {rest_length}")

        # ---------------------------------------------------------
        # Current distance
        # ---------------------------------------------------------

        distance = cmds.createNode(
            "distanceBetween",
            name=f"{name}_distance_DB",
        )

        # Capture root joint's world matrix at build time.
        root_world_matrix = cmds.getAttr(f"{root_joint}.worldMatrix[0]")

        # Store the matrix as a static value.
        cmds.setAttr(
            f"{distance}.inMatrix1",
            *root_world_matrix,
            type="matrix",
        )

        cmds.connectAttr(
            f"{ik_handle}.worldMatrix[0]",
            f"{distance}.inMatrix2",
        )

        # ---------------------------------------------------------
        # Current distance / rest distance
        # ---------------------------------------------------------

        ratio = cmds.createNode(
            "multiplyDivide",
            name=f"{name}_length_ratio_MD",
        )

        cmds.setAttr(
            f"{ratio}.operation",
            2,  # Divide #type:ignore
        )

        cmds.connectAttr(
            f"{distance}.distance",
            f"{ratio}.input1X",
        )

        cmds.setAttr(
            f"{ratio}.input2X",
            rest_length,  # type:ignore
        )

        # ---------------------------------------------------------
        # Select stretch vs compression multiplier
        # ---------------------------------------------------------

        condition = cmds.createNode(
            "condition",
            name=f"{name}_length_mode_CON",
        )

        cmds.setAttr(
            f"{condition}.operation",
            2,  # Greater than #type:ignore
        )

        cmds.connectAttr(
            f"{ratio}.outputX",
            f"{condition}.firstTerm",
        )

        cmds.setAttr(
            f"{condition}.secondTerm",
            1.0,  # type:ignore
        )

        cmds.connectAttr(
            f"{control}.stretch",
            f"{condition}.colorIfTrueR",
        )

        cmds.connectAttr(
            f"{control}.compress",
            f"{condition}.colorIfFalseR",
        )

        # ---------------------------------------------------------
        # Blend between rest ratio (1) and current ratio
        # ---------------------------------------------------------

        blend = cmds.createNode(
            "blendColors",
            name=f"{name}_length_blend_BLC",
        )

        cmds.connectAttr(
            f"{condition}.outColorR",
            f"{blend}.blender",
        )

        cmds.connectAttr(
            f"{ratio}.outputX",
            f"{blend}.color1R",
        )

        cmds.setAttr(
            f"{blend}.color2R",
            1.0,  # type:ignore
        )

        # ---------------------------------------------------------
        # Store original length on root joint
        # ---------------------------------------------------------

        if not cmds.attributeQuery(
            "restLength",
            node=root_joint,
            exists=True,
        ):
            cmds.addAttr(
                root_joint,
                longName="restLength",
                attributeType="double",
                defaultValue=rest_length,
            )

        cmds.setAttr(
            f"{root_joint}.restLength",
            rest_length,  # type:ignore
        )

        cmds.setAttr(
            f"{root_joint}.restLength",
            lock=True,
        )

        # ---------------------------------------------------------
        # Apply length multiplier
        # ---------------------------------------------------------

        length_mult = cmds.createNode(
            "multiplyDivide",
            name=f"{name}_length_MD",
        )

        cmds.setAttr(
            f"{length_mult}.operation",
            1,  # Multiply #type:ignore
        )

        cmds.connectAttr(
            f"{blend}.outputR",
            f"{length_mult}.input1X",
        )

        # Use the original signed local joint translation.
        # This supports joints oriented along different axes.
        translate = cmds.getAttr(f"{end_joint}.translate")[0]

        axis = max(
            "XYZ",
            key=lambda a: abs(translate["XYZ".index(a)]),
        )

        original_value = translate["XYZ".index(axis)]

        cmds.setAttr(
            f"{length_mult}.input2X",
            original_value,
        )

        cmds.connectAttr(
            f"{length_mult}.outputX",
            f"{end_joint}.translate{axis}",
            force=True,
        )

    def add_blink_length_attributes(self) -> None:
        """Add stretch/compression controls to both blink controls."""

        for vertical in ("upper", "lower"):
            control = self.blink_controls[vertical].transform

            for attr in ("stretch", "compress"):
                if cmds.attributeQuery(
                    attr,
                    node=control,
                    exists=True,
                ):
                    continue

                cmds.addAttr(
                    control,
                    longName=attr,
                    attributeType="double",
                    minValue=0.0,
                    maxValue=1.0,
                    defaultValue=0.0,
                    keyable=True,
                )

    def build_eyelid_ik(self, sub_divisions: int = 11, major_division: int = 7) -> None:
        """
        Build single-chain IK joints for the upper/lower eyelids.

        Each chain:
            - Starts at the eye center.
            - Ends at a sampled point on the eyelid.
            - Has an IK handle pinned to the blendshape curve.

        All IK chains are stored under the hidden eye setup group.

        Requires:
            self.setup_blink_curves()
            self.build_blink_blendshapes()
        """

        if sub_divisions < 2:
            raise ValueError("sub_divisions must be >= 2")

        # -----------------------------------------------------
        # Setup hierarchy
        # -----------------------------------------------------

        self.eye_ik_grp = create_transform(
            name=f"eye_ik_{self.side}_grp",
            parent=self.blink_setup_grp,
        )

        self.collision_grp = create_transform(
            name=f"eyelid_collision_{self.side}_grp",
            parent=self.blink_setup_grp,
        )

        self.eyelid_ik = {
            "upper": [],
            "lower": [],
        }

        # -----------------------------------------------------
        # Eye center position
        # -----------------------------------------------------

        center = cmds.xform(
            self.guides["center_piv"],
            query=True,
            worldSpace=True,
            translation=True,
        )

        # -----------------------------------------------------
        # Build upper/lower chains
        # -----------------------------------------------------

        for vertical in ("upper", "lower"):
            vertical_grp = create_transform(
                name=f"{vertical}_ik_{self.side}_grp",
                parent=self.eye_ik_grp,
            )

            rest_curve = self.blink_guides[f"eyelid_{vertical}_rest"]

            animated_curve = self.blink_curves[vertical]

            for i in range(sub_divisions):
                percent = i / (sub_divisions - 1)

                name = f"{vertical}_eyelid_{i:02d}_{self.side}"

                # ---------------------------------------------
                # Initial position on neutral/rest curve
                # ---------------------------------------------

                # ---------------------------------------------
                # Initial position and orientation
                # ---------------------------------------------

                position = cmds.pointOnCurve(
                    rest_curve,
                    parameter=percent,
                    turnOnPercentage=True,
                    position=True,
                )

                # Use the same matrix as the sub-control.
                control_matrix = self.get_curve_point_matrix(
                    curve=self.guides[f"eyelid_{vertical}_rest"],
                    parameter=percent,
                    tangent_orient=self.orient_sub_expression_controls,
                )

                # Root joint stays at the eye center.
                root_matrix = MMatrix(control_matrix)

                root_values = list(root_matrix)
                root_values[12] = center[0]  # type:ignore
                root_values[13] = center[1]  # type:ignore
                root_values[14] = center[2]  # type:ignore

                root_matrix = MMatrix(root_values)

                # ---------------------------------------------
                # Create root joint
                # ---------------------------------------------

                root_joint = create_joint(
                    name=f"{name}_root",
                    parent=vertical_grp,
                    transform=root_matrix,
                    connect=False,
                )

                # ---------------------------------------------
                # Create end joint
                # ---------------------------------------------

                end_joint = create_joint(
                    name=f"{name}_end",
                    parent=root_joint,
                    transform=control_matrix,
                    connect=False,
                )

                # ---------------------------------------------
                # Create single-chain IK
                # ---------------------------------------------

                ik = create_ik_single_chain(
                    name=name,
                    start_joint=root_joint,
                    end_joint=end_joint,
                )

                cmds.parent(
                    ik.handle,
                    vertical_grp,
                )

                # ---------------------------------------------
                # Create curve pin
                # ---------------------------------------------

                pin = create_transform(
                    name=f"{name}_pin",
                    parent=vertical_grp,
                )

                collision = create_transform(
                    name=f"{name}_collision",
                    parent=self.collision_grp,
                    transform=pin,
                )

                cmds.xform(
                    pin,
                    worldSpace=True,
                    translation=position,  # type:ignore
                )

                self.pin_transform_to_curve(
                    curve=animated_curve,
                    transform=pin,
                    parameter=percent,
                )

                # ---------------------------------------------
                # Pin IK handle to animated curve
                # ---------------------------------------------

                cmds.pointConstraint(
                    collision,
                    ik.handle,
                    maintainOffset=False,
                )

                # ---------------------------------------------
                # Store results
                # ---------------------------------------------

                # -----------------------------------------------------
                # Rest length
                # -----------------------------------------------------

                rest_length = sum((position[i] - center[i]) ** 2 for i in range(3)) ** 0.5  # type:ignore

                # -----------------------------------------------------
                # Stretch/compression
                # -----------------------------------------------------

                """self.connect_ik_stretch_compress(
                    name=name,
                    root_joint=root_joint,
                    end_joint=end_joint,
                    ik_handle=ik.handle,
                    control=self.blink_controls[vertical].transform,
                    rest_length=rest_length,
                )"""

                self.eyelid_ik[vertical].append(
                    {
                        "root_joint": root_joint,
                        "end_joint": end_joint,
                        "ik": ik,
                        "pin": pin,
                        "parameter": percent,
                        "collision": collision,
                    }
                )

    def pin_transform_to_curve(
        self,
        curve: str,
        transform: str,
        parameter: float,
    ) -> str:
        """
        Pin a transform to a normalized arc-length position
        on a deforming NURBS curve.

        Uses pointOnCurveInfo and parent-space conversion.

        Args:
            curve: Deforming NURBS curve.
            transform: Transform to drive.
            parameter: Normalized position from 0 to 1.

        Returns:
            Created pointOnCurveInfo node.
        """

        shapes = (
            cmds.listRelatives(
                curve,
                shapes=True,
                noIntermediate=True,
                type="nurbsCurve",
                fullPath=True,
            )
            or []
        )

        if not shapes:
            raise ValueError(f"No NURBS curve found: {curve}")

        shape = shapes[0]

        # -----------------------------------------------------
        # Evaluate curve position
        # -----------------------------------------------------

        poc = cmds.createNode(
            "pointOnCurveInfo",
            name=f"{transform}_POC",
        )

        cmds.connectAttr(
            f"{shape}.worldSpace[0]",
            f"{poc}.inputCurve",
        )

        cmds.setAttr(
            f"{poc}.turnOnPercentage",
            True,  # type:ignore
        )

        cmds.setAttr(
            f"{poc}.parameter",
            parameter,  # type:ignore
        )

        # -----------------------------------------------------
        # Convert world position to parent-local position
        # -----------------------------------------------------

        compose = cmds.createNode(
            "composeMatrix",
            name=f"{transform}_CM",
        )

        cmds.connectAttr(
            f"{poc}.position",
            f"{compose}.inputTranslate",
        )

        mult = cmds.createNode(
            "multMatrix",
            name=f"{transform}_MM",
        )

        cmds.connectAttr(
            f"{compose}.outputMatrix",
            f"{mult}.matrixIn[0]",
        )

        cmds.connectAttr(
            f"{transform}.parentInverseMatrix[0]",
            f"{mult}.matrixIn[1]",
        )

        decompose = cmds.createNode(
            "decomposeMatrix",
            name=f"{transform}_DCM",
        )

        cmds.connectAttr(
            f"{mult}.matrixSum",
            f"{decompose}.inputMatrix",
        )

        cmds.connectAttr(
            f"{decompose}.outputTranslate",
            f"{transform}.translate",
        )

        return poc

    def get_curve_mid_y(self, curve: str) -> float:
        """Get the world-space Y of a curve's arc-length midpoint."""
        matrix = self.get_curve_point_matrix(
            curve=curve,
            parameter=0.5,
            tangent_orient=False,
        )
        return matrix[13]

    def create_blink_remap(
        self,
        name: str,
        input_attr: str,
        positions: list[tuple[float, float]],
    ) -> str:
        """
        Build a piecewise-linear remapValue node.

        positions: [(input_position, output_weight), ...]
        """
        positions = sorted(positions, key=lambda item: item[0])

        node = cmds.createNode("remapValue", name=name)

        cmds.connectAttr(input_attr, f"{node}.inputValue")

        cmds.setAttr(f"{node}.inputMin", positions[0][0])  # type:ignore
        cmds.setAttr(f"{node}.inputMax", positions[-1][0])  # type:ignore
        cmds.setAttr(f"{node}.outputMin", 0)  # type:ignore
        cmds.setAttr(f"{node}.outputMax", 1)  # type:ignore

        minimum = positions[0][0]
        span = positions[-1][0] - minimum

        if span <= 1e-8:
            raise ValueError(f"Invalid remap range: {name}")

        for i, (position, weight) in enumerate(positions):
            normalized = (position - minimum) / span

            cmds.setAttr(
                f"{node}.value[{i}].value_Position",
                normalized,  # type:ignore
            )
            cmds.setAttr(
                f"{node}.value[{i}].value_FloatValue",
                weight,  # type:ignore
            )
            # Linear interpolation
            cmds.setAttr(
                f"{node}.value[{i}].value_Interp",
                1,  # type:ignore
            )

        return f"{node}.outValue"

    def get_expression_placement_matrix(
        self,
        vertical: str,
        index: int,
        parameter: float,
        curve: str,
    ) -> MMatrix:
        """Get expression control placement from an explicit guide or curve."""

        # Only interior macro controls can use explicit guides.
        if index in (1, 2, 3):
            guide_name = f"{vertical}_macro_{index:02d}_{self.side}"

            # Support either guide names or guide objects stored in self.guides.
            guide = self.guides.get(guide_name)

            if guide and cmds.objExists(guide):
                return MMatrix(
                    cmds.xform(
                        guide,
                        query=True,
                        worldSpace=True,
                        matrix=True,
                    )
                )

        # Existing curve-based fallback.
        return self.get_curve_point_matrix(
            curve=curve,
            parameter=parameter,
            tangent_orient=self.orient_sub_expression_controls,
        )

    def build_blink_blendshapes(
        self,
        extension: float = 2.0,
    ) -> None:
        """
        Build upper and lower eyelid blendshape systems.

        Upper eyelid:
            upblink -> upper_rest -> blink -> downblink

        Lower eyelid:
            upblink -> blink -> lower_rest -> downblink

        translateY = 0:
            All blendshape weights are zero.

        Moving toward the outer target:
            Weight increases from 0 to 1 at the guide position.
            Continues extrapolating up to `extension`.

        Moving toward the opposite side:
            The blink target activates first.
            Then fades out as the final target activates.

        Requires:
            self.setup_blink_curves() to have been called.
            self.blink_controls["upper"/"lower"] to exist.
        """

        if extension < 1.0:
            raise ValueError("extension must be >= 1.0")

        # ---------------------------------------------------------
        # Guide midpoint heights
        # ---------------------------------------------------------

        guide_keys = [
            "eyelid_upblink",
            "eyelid_upper_rest",
            "eyelid_blink",
            "eyelid_lower_rest",
            "eyelid_downblink",
        ]

        heights = {key: self.get_curve_mid_y(self.blink_guides[key]) for key in guide_keys}

        up_y = heights["eyelid_upblink"]
        upper_y = heights["eyelid_upper_rest"]
        blink_y = heights["eyelid_blink"]
        lower_y = heights["eyelid_lower_rest"]
        down_y = heights["eyelid_downblink"]

        if not (up_y > upper_y > blink_y > lower_y > down_y):
            raise ValueError(
                "Eyelid guide midpoint Y positions must follow:\n"
                "upblink > upper_rest > blink > lower_rest > downblink\n"
                f"Received: {heights}"
            )

        # ---------------------------------------------------------
        # Initialize storage
        # ---------------------------------------------------------

        self.blink_blendshapes = {}

        configurations = {
            "upper": {
                "rest": "eyelid_upper_rest",
                "targets": [
                    "eyelid_upblink",
                    "eyelid_blink",
                    "eyelid_downblink",
                ],
            },
            "lower": {
                "rest": "eyelid_lower_rest",
                "targets": [
                    "eyelid_upblink",
                    "eyelid_blink",
                    "eyelid_downblink",
                ],
            },
        }

        # ---------------------------------------------------------
        # Build each eyelid
        # ---------------------------------------------------------

        for vertical, config in configurations.items():
            rest_key = config["rest"]
            rest_y = heights[rest_key]  # type:ignore

            # Use duplicated neutral curve from setup_blink_curves().
            curve = self.blink_curves[vertical]

            # Use duplicated targets, not original guides.
            targets = [
                *[self.blink_guides[key] for key in config["targets"]],
                self.expression_rest_curves[vertical],
            ]

            # -----------------------------------------------------
            # Create blendShape
            # -----------------------------------------------------

            blendshape = cmds.blendShape(
                *targets,
                curve,
                name=f"{vertical}_blink_{self.side}_BS",
                frontOfChain=True,
            )[0]  # type:ignore

            # Allow weights greater than 1 for extrapolation.
            cmds.setAttr(
                f"{blendshape}.weight",
                size=len(targets),
            )

            expression_index = len(targets) - 1

            cmds.setAttr(
                f"{blendshape}.weight[{expression_index}]",
                1.0,  # type:ignore
            )

            # -----------------------------------------------------
            # Calculate relative Y distances
            # -----------------------------------------------------

            up_pos = up_y - rest_y
            blink_pos = blink_y - rest_y
            down_pos = down_y - rest_y

            control = self.blink_controls[vertical].transform
            input_attr = f"{control}.translateY"

            # -----------------------------------------------------
            # Piecewise weight mappings
            #
            # Each point is:
            #     (translateY, blendshape_weight)
            #
            # The remap helper normalizes the input positions
            # and interpolates linearly between them.
            # -----------------------------------------------------

            if vertical == "upper":
                # Rest -> upblink -> extended upblink
                up_points = [
                    (down_pos, 0.0),
                    (0.0, 0.0),
                    (up_pos, 1.0),
                    (up_pos * extension, extension),
                ]

                # Rest -> blink -> downblink
                blink_points = [
                    (down_pos, 0.0),
                    (blink_pos, 1.0),
                    (0.0, 0.0),
                    (up_pos * extension, 0.0),
                ]

                # Blink -> downblink
                down_points = [
                    (down_pos * extension, extension),
                    (blink_pos, 0.0),
                    (0.0, 0.0),
                    (up_pos * extension, 0.0),
                ]

            else:
                # Blink -> upblink
                up_points = [
                    (down_pos * extension, 0.0),
                    (0.0, 0.0),
                    (blink_pos, 0.0),
                    (up_pos * extension, extension),
                ]

                # Rest -> blink -> upblink
                blink_points = [
                    (down_pos * extension, 0.0),
                    (0.0, 0.0),
                    (blink_pos, 1.0),
                    (up_pos, 0.0),
                ]

                # Rest -> downblink -> extended downblink
                down_points = [
                    (down_pos * extension, extension),
                    (down_pos, 1.0),
                    (0.0, 0.0),
                    (up_pos, 0.0),
                ]

            mappings = [
                ("upblink", up_points),
                ("blink", blink_points),
                ("downblink", down_points),
            ]

            # -----------------------------------------------------
            # Create remap nodes and connect blendshape weights
            # -----------------------------------------------------

            for index, (target_name, points) in enumerate(mappings):
                # remapValue output is normalized to 0..1.
                # Scale back to the desired maximum weight.
                maximum_weight = max(weight for _, weight in points)

                normalized_points = [
                    (position, weight / maximum_weight) for position, weight in points
                ]

                output = self.create_blink_remap(
                    name=f"{vertical}_{target_name}_{self.side}_RMV",
                    input_attr=input_attr,
                    positions=normalized_points,
                )

                multiplier = cmds.createNode(
                    "multiplyDivide",
                    name=f"{vertical}_{target_name}_{self.side}_MD",
                )

                cmds.setAttr(
                    f"{multiplier}.input2X",
                    maximum_weight,  # type:ignore
                )

                cmds.connectAttr(
                    output,
                    f"{multiplier}.input1X",
                )

                cmds.connectAttr(
                    f"{multiplier}.outputX",
                    f"{blendshape}.weight[{index}]",
                    force=True,
                )

            # -----------------------------------------------------
            # Store results
            # -----------------------------------------------------

            self.blink_blendshapes[vertical] = blendshape

    def setup_blink_curves(self) -> None:
        """
        Create the hidden eyelid blendshape setup.

        Hierarchy:
            rig
                eyes_setup
                    eye_blink_L_grp
                        duplicated guides
                        upper_blink_curve_L
                        lower_blink_curve_L

        The original guide curves remain untouched.
        """

        # ----------------------------------------
        # Shared setup group
        # ----------------------------------------

        if not cmds.objExists(self.setup_parent):
            raise ValueError(f"Eye setup parent does not exist: {self.setup_parent}")

        if cmds.objExists(self.setup_group):
            self.eyes_setup_grp = self.setup_group
        else:
            self.eyes_setup_grp = create_transform(
                name=self.setup_group,
                parent=self.setup_parent,
            )

        # ----------------------------------------
        # Side-specific group
        # ----------------------------------------

        side_group_name = f"eye_blink_{self.side}_grp"

        if cmds.objExists(side_group_name):
            raise ValueError(f"Blink setup already exists: {side_group_name}")

        self.blink_setup_grp = create_transform(
            name=side_group_name,
            parent=self.eyes_setup_grp,
        )

        # ----------------------------------------
        # Duplicate all five guide curves
        # ----------------------------------------

        guide_keys = [
            "eyelid_upblink",
            "eyelid_upper_rest",
            "eyelid_blink",
            "eyelid_lower_rest",
            "eyelid_downblink",
        ]

        for key in guide_keys:
            source_curve = self.guides[key]

            if not cmds.objExists(source_curve):
                raise ValueError(f"Missing eyelid guide: {source_curve}")

            duplicate_name = f"{key}_{self.side}_target"

            duplicate = cmds.duplicate(
                source_curve,
                name=duplicate_name,
                returnRootsOnly=True,
            )[0]

            cmds.parent(
                duplicate,
                self.blink_setup_grp,
            )

            self.blink_guides[key] = duplicate

        # ----------------------------------------
        # Duplicate the two neutral/rest curves
        # ----------------------------------------

        for vertical in ("upper", "lower"):
            rest_key = f"eyelid_{vertical}_rest"

            curve = cmds.duplicate(
                self.blink_guides[rest_key],
                name=f"{vertical}_blink_curve_{self.side}",
                returnRootsOnly=True,
            )[0]

            cmds.parent(
                curve,
                self.blink_setup_grp,
            )

            self.blink_curves[vertical] = curve

        # ----------------------------------------
        # Hide shared setup hierarchy
        # ----------------------------------------

        self.expression_rest_curves = {}

        for vertical in ("upper", "lower"):
            rest_key = f"eyelid_{vertical}_rest"

            curve = cmds.duplicate(
                self.blink_guides[rest_key],
                name=f"{vertical}_expression_rest_{self.side}",
                returnRootsOnly=True,
            )[0]

            cmds.parent(curve, self.blink_setup_grp)

            self.expression_rest_curves[vertical] = curve

        cmds.setAttr(
            f"{self.eyes_setup_grp}.visibility",
            False,  # type:ignore
        )

    def get_curve_point_matrix(
        self,
        curve: str,
        parameter: float,
        tangent_orient: bool = True,
        up_vector: tuple[float, float, float] = (0, 1, 0),
        aim_axis: str = "x",
        mirror: bool = False,
        mirror_axis: str = "x",
    ) -> MMatrix:
        """
        Get a world-space matrix at a normalized curve parameter.

        When mirror=True:
            1. Reflect the curve position and tangent into mirrored space.
            2. Construct the orientation in mirrored space.
            3. Reflect the completed matrix back across the mirror axis.

        The resulting position matches the original curve, while
        the orientation follows the mirrored construction.

        Args:
            curve: NURBS curve transform or shape.
            parameter: Normalized arc-length parameter (0.0-1.0).
            tangent_orient: Orient along the curve tangent.
            up_vector: World-space reference up vector.
            aim_axis: Local axis aligned with tangent ('x', 'y', 'z').
            mirror: Enable mirrored orientation construction.
            mirror_axis: World reflection axis ('x', 'y', 'z').

        Returns:
            World-space MMatrix.
        """

        # ---------------------------------------------------------
        # Resolve curve shape
        # ---------------------------------------------------------

        if cmds.nodeType(curve) == "nurbsCurve":
            shape = curve
        else:
            shapes = (
                cmds.listRelatives(
                    curve,
                    shapes=True,
                    noIntermediate=True,
                    type="nurbsCurve",
                    fullPath=True,
                )
                or []
            )

            if not shapes:
                raise ValueError(f"No NURBS curve found: {curve}")

            shape = shapes[0]

        selection = MSelectionList()
        selection.add(shape)

        dag_path = selection.getDagPath(0)
        fn_curve = MFnNurbsCurve(dag_path)

        # ---------------------------------------------------------
        # Sample curve
        # ---------------------------------------------------------

        parameter = max(0.0, min(1.0, parameter))

        length = fn_curve.length()
        curve_param = fn_curve.findParamFromLength(length * parameter)

        position = fn_curve.getPointAtParam(
            curve_param,
            MSpace.kWorld,
        )

        tangent = fn_curve.tangent(
            curve_param,
            MSpace.kWorld,
        ).normal()

        # ---------------------------------------------------------
        # Reflection helpers
        # ---------------------------------------------------------

        axis = mirror_axis.lower()

        if axis not in ("x", "y", "z"):
            raise ValueError("mirror_axis must be 'x', 'y', or 'z'")

        axis_index = {"x": 0, "y": 1, "z": 2}[axis]

        def reflect(vector: MVector) -> MVector:
            values = [vector.x, vector.y, vector.z]
            values[axis_index] *= -1
            return MVector(*values)

        # ---------------------------------------------------------
        # Mirror curve sample BEFORE constructing orientation
        # ---------------------------------------------------------

        if mirror:
            position = MPoint(
                *reflect(
                    MVector(
                        position.x,
                        position.y,
                        position.z,
                    )
                )
            )

            tangent = reflect(tangent)

        # ---------------------------------------------------------
        # Construct orientation
        # ---------------------------------------------------------

        if tangent_orient:
            aim = tangent.normal()
            up = MVector(*up_vector).normal()

            # Avoid parallel aim/up vectors.
            if abs(aim * up) > 0.999:
                fallback = MVector(0, 0, 1) if abs(aim.z) < 0.999 else MVector(1, 0, 0)
                up = fallback

            side = (aim ^ up).normal()
            up = (side ^ aim).normal()

            if aim_axis.lower() == "x":
                x_axis = aim
                y_axis = up
                z_axis = side

            elif aim_axis.lower() == "y":
                x_axis = -side
                y_axis = aim
                z_axis = up

            elif aim_axis.lower() == "z":
                x_axis = up
                y_axis = -side
                z_axis = aim

            else:
                raise ValueError("aim_axis must be 'x', 'y', or 'z'")

        else:
            x_axis = MVector(1, 0, 0)
            y_axis = MVector(0, 1, 0)
            z_axis = MVector(0, 0, 1)

        # ---------------------------------------------------------
        # Build matrix in sampled space
        # ---------------------------------------------------------

        matrix = MMatrix(
            [
                x_axis.x,
                x_axis.y,
                x_axis.z,
                0.0,
                y_axis.x,
                y_axis.y,
                y_axis.z,
                0.0,
                z_axis.x,
                z_axis.y,
                z_axis.z,
                0.0,
                position.x,
                position.y,
                position.z,
                1.0,
            ]
        )

        # ---------------------------------------------------------
        # Reflect completed matrix back across world axis
        # ---------------------------------------------------------

        if mirror:
            scale = [1.0, 1.0, 1.0]
            scale[axis_index] = -1.0

            mirror_matrix = MMatrix(
                [
                    scale[0],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    scale[1],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    scale[2],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )

            # Maya uses row-vector matrix convention.
            # Reflect the world-space matrix back.
            matrix = matrix * mirror_matrix

        return matrix

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

    def setup_expression_follow(self) -> None:
        """Create the fake eyelid follow group before expression controls."""
        expression_follow_offset = create_transform(
            name=f"eyelid_follow_{self.side}_fake_grp_offset",
            parent=self.eyes_setup_grp,
            transform=self.guides["center_piv"],
        )

        self.expression_follow_grp = create_transform(
            name=f"eyelid_follow_{self.side}_fake_grp",
            parent=expression_follow_offset,
            transform=self.guides["center_piv"],
        )

        cmds.connectAttr(
            f"{self.look_offset}.rotate",
            f"{self.expression_follow_grp}.rotate",
            force=True,
        )

    def build(
        self,
        sub_divisions: int = 11,
    ) -> None:
        #######
        # Set up look follow
        #######

        self.sub_eyelid_controls = {
            "upper": [],
            "lower": [],
        }
        self.sub_joints = []

        self.look_offset = create_transform(
            name=f"look_offset_{self.side}",
            parent=self.main_ctrl,
            transform=self.guides["center_piv"],
        )

        self.sub_grp = create_transform(
            name=f"sub_eyelid_controls_{self.side}",
            parent=self.main_ctrl,
            transform=self.guides["center_piv"],
        )

        ##########
        # Main Control Behavior
        ##########

        self.blink_controls = {}
        for vertical in ["upper", "lower"]:
            side_mod = -1 if self.side == "R" else 1
            blink_matrix = self.get_curve_point_matrix(
                curve=self.guides[f"eyelid_{vertical}_rest"],
                parameter=0.5,
                tangent_orient=False,
                mirror=self.side == "R",
            )

            self.blink_controls[vertical] = create_control(
                name=f"{vertical}_blink_{self.side}",
                parent=self.main_ctrl,
                transform=blink_matrix,
                size=self.control_size,
                control_shape=ControlShape.SEMI_CIRCLE,
                direction="z",
                dimensions=(1, 1, 1 if vertical == "upper" else -1),
                position_offset=(0, 0, 1),
            )

        # Build hidden setup hierarchy and duplicate guides
        self.setup_blink_curves()

        # Connect blink controls to blendshape curves
        self.build_blink_blendshapes(extension=2.0)

        # self.add_blink_length_attributes()

        self.build_eyelid_ik(
            sub_divisions=sub_divisions,
        )

        self.build_eyelid_collision(
            push=0.5,
        )

        for vertical in ["upper", "lower"]:
            joints = []
            for i in range(sub_divisions):
                percent = i / (sub_divisions - 1)
                matrix = self.get_curve_point_matrix(
                    curve=self.guides[f"eyelid_{vertical}_rest"],
                    parameter=percent,
                    tangent_orient=True,
                )

                sub_ctrl = create_control(
                    name=f"{vertical}_{i}_{self.side}",
                    parent=self.sub_grp,
                    transform=matrix,
                    size=self.control_size / 10,
                    control_shape="circle",
                    direction="z",
                )
                sub_jnt = create_joint(
                    name=f"{vertical}_{i}_{self.side}",
                    parent=self.joint_parent,
                    transform=sub_ctrl.transform,
                )

                self.sub_eyelid_controls[vertical].append(sub_ctrl)
                self.sub_joints.append(sub_jnt)
                joints.append(sub_jnt)

            lead_name = f"inner_corner_{vertical}_eyelid_{self.side}_jnt"

            lead_joint = cmds.rename(
                joints[0],
                lead_name,
            )

            # Update stored reference after renaming.
            joints[0] = lead_joint

            # Tag the lead influence and its split targets.
            tag_for_weight_split(
                influence=lead_joint,
                split_influences=joints,
            )

        # After sub-controls have been created
        self.connect_sub_controls_to_ik()

        # Build expression control hierarchies.

        self.build_expression_controls()
        # self.setup_expression_follow()

        self.build_macro_joints()
        self.skin_macro_curves()
        self.smooth_macro_curve_weights()
        self.connect_major_controls()
        self.setup_sub_eyelid_visibility()
