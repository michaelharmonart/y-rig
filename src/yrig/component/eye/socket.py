import math

from maya import cmds
from maya.api.OpenMaya import MEulerRotation, MMatrix, MSpace, MTransformationMatrix, MVector

from yrig.control import Control, create_control
from yrig.joint import create_joint
from yrig.skin.split.tag import tag_for_weight_split
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.transform import create_transform

from .guide_curve import GuideCurve


class Socket:
    def __init__(
        self,
        guides: dict,
        side: str,
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

    def create_socket_follow(
        self,
        default_mult: float = 0.5,
    ) -> str:
        """
        Create a Workshop-style blended driver offset
        for the socket controls.

        0.0 = No follow
        0.5 = Half follow
        1.0 = Full follow

        The follow hierarchy is positioned at the
        main eye control's pivot.
        """

        driver = self.main_ctrl

        # -------------------------------------------------
        # Follow attribute
        # -------------------------------------------------

        if not cmds.attributeQuery(
            "socket_follow",
            node=driver,
            exists=True,
        ):
            cmds.addAttr(
                driver,
                longName="socket_follow",
                attributeType="double",
                minValue=0.0,
                maxValue=1.0,
                defaultValue=default_mult,
                keyable=True,
            )

        # -------------------------------------------------
        # Create driver hierarchy
        # -------------------------------------------------

        # IMPORTANT:
        # control_grp must not inherit transforms
        # from the main eye control.

        driver_pos = create_transform(
            name=f"socket_{self.side}_follow_pos",
            parent=self.control_grp,
            transform=driver,
        )

        driver_offset = create_transform(
            name=f"socket_{self.side}_follow_offset",
            parent=driver_pos,
            transform=driver,
        )

        # -------------------------------------------------
        # Driver relative to parent space
        # -------------------------------------------------

        relative_matrix = cmds.createNode(
            "multMatrix",
            name=f"socket_{self.side}_follow_relative_mm",
        )

        cmds.connectAttr(
            f"{driver}.worldMatrix[0]",
            f"{relative_matrix}.matrixIn[0]",
        )

        cmds.connectAttr(
            f"{self.control_grp}.worldInverseMatrix[0]",
            f"{relative_matrix}.matrixIn[1]",
        )

        # -------------------------------------------------
        # Store rest relative matrix
        # -------------------------------------------------

        rest_matrix = MMatrix(cmds.getAttr(f"{relative_matrix}.matrixSum"))

        rest_inverse = cmds.createNode(
            "inverseMatrix",
            name=f"socket_{self.side}_follow_rest_inverse",
        )

        cmds.setAttr(
            f"{rest_inverse}.inputMatrix",
            *list(rest_matrix),
            type="matrix",
        )

        # -------------------------------------------------
        # Calculate driver delta
        # -------------------------------------------------

        delta_matrix = cmds.createNode(
            "multMatrix",
            name=f"socket_{self.side}_follow_delta_mm",
        )

        cmds.connectAttr(
            f"{relative_matrix}.matrixSum",
            f"{delta_matrix}.matrixIn[0]",
        )

        cmds.connectAttr(
            f"{rest_inverse}.outputMatrix",
            f"{delta_matrix}.matrixIn[1]",
        )

        # -------------------------------------------------
        # Blend identity -> driver delta
        # -------------------------------------------------

        blend_matrix = cmds.createNode(
            "blendMatrix",
            name=f"socket_{self.side}_follow_bm",
        )

        cmds.connectAttr(
            f"{delta_matrix}.matrixSum",
            f"{blend_matrix}.target[0].targetMatrix",
        )

        cmds.connectAttr(
            f"{driver}.socket_follow",
            f"{blend_matrix}.target[0].weight",
        )

        # -------------------------------------------------
        # Drive follow offset
        # -------------------------------------------------

        cmds.connectAttr(
            f"{blend_matrix}.outputMatrix",
            f"{driver_offset}.offsetParentMatrix",
        )

        self.socket_follow_pos = driver_pos
        self.socket_follow_offset = driver_offset
        self.socket_follow_blend = blend_matrix

        return driver_offset

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

    def create_socket_spline_follow(
        self,
        control: Control,
        name: str,
    ) -> str:
        """
        Insert a spline-follow group above a socket control's NPO.

        The follow group is initialized at the control's
        existing parent position, not the control position.

        This preserves the NPO's original rest transform.
        """

        npo = control.offset

        parent = cmds.listRelatives(
            npo,
            parent=True,
            fullPath=True,
        )[0]

        follow = create_transform(
            name=f"{name}_spline_follow",
            parent=parent,
            transform=parent,
        )

        cmds.parent(
            npo,
            follow,
        )

        return follow

    def sort_transforms_center_out(
        self,
        transforms: list[str],
    ) -> list[str]:
        """
        Sort transforms from the character center outward
        along world X.

        Works for both positive and negative X.

        Examples:
            Left:   X = 1, 3, 5, 7
            Right:  X = -1, -3, -5, -7

        Returns:
            A new sorted list of transform names.
        """

        return sorted(
            transforms,
            key=lambda transform: abs(
                cmds.xform(
                    transform,
                    query=True,
                    worldSpace=True,
                    translation=True,
                )[0]  # type:ignore
            ),
        )

    def connect_socket_spline_delta(
        self,
        pin: str,
        control: Control,
        rest_matrix: list[float],
        name: str,
    ) -> None:
        """
        Drive a socket control NPO from a spline pin,
        preserving the original rest position.

        The control retains its existing parent hierarchy
        and animator-facing local transforms.
        """

        pin_rest = MMatrix(
            cmds.xform(
                pin,
                query=True,
                worldSpace=True,
                matrix=True,
            )
        )

    def connect_socket_spline_follow(
        self,
        pin: str,
        control: Control,
    ) -> None:
        """
        Drive a socket control's NPO from a spline pin.

        Preserve:
            - Original NPO rest position
            - Existing upper/lower parent hierarchy
            - Animator-facing control channels

        The spline is evaluated in world space and
        converted into the NPO's parent space.
        """

        npo = control.offset

        # ----------------------------------------
        # Capture rest matrices
        # ----------------------------------------

        pin_rest = MMatrix(
            cmds.xform(
                pin,
                query=True,
                worldSpace=True,
                matrix=True,
            )
        )

        npo_rest = MMatrix(
            cmds.xform(
                npo,
                query=True,
                worldSpace=True,
                matrix=True,
            )
        )

        # Existing local channel matrix.
        local_rest = MMatrix(cmds.getAttr(f"{npo}.matrix"))

        # Difference between the original NPO
        # position and spline's initial position.
        rest_offset = npo_rest * pin_rest.inverse()

        # ----------------------------------------
        # Build matrix network
        # ----------------------------------------

        mm = cmds.createNode(
            "multMatrix",
            name=f"{npo}_spline_mm",
        )

        # Maya row-vector convention:
        #
        # OPM = local^-1
        #       * restOffset
        #       * pinWorld
        #       * parentWorld^-1

        cmds.setAttr(
            f"{mm}.matrixIn[0]",
            *list(local_rest.inverse()),
            type="matrix",
        )

        cmds.setAttr(
            f"{mm}.matrixIn[1]",
            *list(rest_offset),
            type="matrix",
        )

        cmds.connectAttr(
            f"{pin}.worldMatrix[0]",
            f"{mm}.matrixIn[2]",
        )

        parent = cmds.listRelatives(
            npo,
            parent=True,
            fullPath=True,
        )

        if parent:
            cmds.connectAttr(
                f"{parent[0]}.worldInverseMatrix[0]",
                f"{mm}.matrixIn[3]",
            )

        cmds.connectAttr(
            f"{mm}.matrixSum",
            f"{npo}.offsetParentMatrix",
            force=True,
        )

    def build_socket(self) -> None:

        self.major_controls = {}
        self.parent_controls = {}
        self.corner_controls = {}
        self.main_joints = {}

        self.sub_socket_control = []
        self.socket_driver_controls = []
        self.socket_splines = {}
        self.socket_pins = {}

        self.socket_follow_grp = self.create_socket_follow(
            default_mult=0.5,
        )

        # -------------------------------------------------
        # Build guide curves
        # -------------------------------------------------

        curve_guides = {}

        for side in ["upper", "lower"]:
            curve_guides[side] = GuideCurve(
                curve=self.guides[f"socket_{side}_curve"],
                resample_amount=7,
                output_names=[
                    f"{side}_inner_corner",
                    f"{side}_inner_01",
                    f"{side}_inner_02",
                    f"{side}_mid",
                    f"{side}_outer_02",
                    f"{side}_outer_01",
                    f"{side}_outer_corner",
                ],
                ignore_handles=True,
                align_normals=True,
                mirror=self.side == "R",
            )

        # -------------------------------------------------
        # Shared corner controls
        # -------------------------------------------------

        for corner, index in [("inner", 0), ("outer", -1)]:
            guide = curve_guides["upper"].locator_list[index]

            control = create_control(
                name=f"socket_{corner}_corner_{self.side}",
                parent=self.socket_follow_grp,
                transform=guide.name,
                size=self.control_size / 4,
                control_shape="circle",
                direction="z",
            )

            self.corner_controls[corner] = control
            self.socket_driver_controls.append(control)

        # -------------------------------------------------
        # Upper and lower driver controls
        # -------------------------------------------------

        for side in ["upper", "lower"]:
            control = create_control(
                name=f"socket_{side}_{self.side}",
                parent=self.socket_follow_grp,
                transform=self.guides[f"socket_mid_{side}"],
                size=self.control_size / 2,
                control_shape="round_square",
                direction="z",
                dimensions=(1, 0.2, 0.2),
            )

            self.parent_controls[f"{side}_ctrl"] = control
            self.socket_driver_controls.append(control)

            cmds.addAttr(
                control.transform,
                longName="sub_socket",
                proxy=f"{self.main_ctrl}.sub_socket",
            )

        # -------------------------------------------------
        # Build intermediate controls and joints
        # -------------------------------------------------

        for side in ["upper", "lower"]:
            guides = curve_guides[side].locator_list

            jnt_list = []
            spline_connections = []

            for i, guide in enumerate(guides):
                if i == 0:
                    control = self.corner_controls["inner"]

                elif i == len(guides) - 1:
                    control = self.corner_controls["outer"]

                else:
                    control = create_control(
                        name=f"{guide.name}_{self.side}",
                        parent=self.parent_controls[f"{side}_ctrl"],
                        transform=guide.name,
                        size=self.control_size / 8,
                        control_shape="circle",
                        direction="z",
                    )

                    self.major_controls[f"{guide.name}_ctrl"] = control
                    self.sub_socket_control.append(control)

                    name = f"{guide.name}_{self.side}"

                    # Independent transform driven by the spline.
                    pin = create_transform(
                        name=f"{name}_spline_pin",
                        parent=self.component_grp,
                        transform=guide.name,
                    )

                    self.socket_pins[name] = pin

                    spline_connections.append((pin, control))

                joint = create_joint(
                    name=f"{guide.name}_{self.side}",
                    transform=control.transform,
                    parent=self.joint_parent,
                )

                self.main_joints[f"{guide.name}_jnt"] = joint
                jnt_list.append(joint)

            # -------------------------------------------------
            # Matrix spline
            # -------------------------------------------------

            driver_list = self.sort_transforms_center_out(
                [
                    self.corner_controls["inner"].transform,
                    self.parent_controls[f"{side}_ctrl"].transform,
                    self.corner_controls["outer"].transform,
                ]
            )

            pinned_transforms = self.sort_transforms_center_out(
                [pin for pin, control in spline_connections]
            )

            self.socket_splines[side] = matrix_spline_from_transforms(
                name=f"socket_{side}_{self.side}",
                pinned_transforms=pinned_transforms,
                cv_transforms=driver_list,
                parent=self.component_grp,
                degree=2,
                stretch=False,
                align_tangent=False,
                interpolate_rotation=False,
                interpolate_scale=False,
            )

            for pin, control in spline_connections:
                self.connect_socket_spline_follow(
                    pin=pin,
                    control=control,
                )

            # -------------------------------------------------
            # Weight splitting
            # -------------------------------------------------

            tag_for_weight_split(
                influence=jnt_list[0],
                split_influences=jnt_list,
            )

        # -------------------------------------------------
        # Cleanup guides
        # -------------------------------------------------

        for guide_curve in curve_guides.values():
            cmds.delete(guide_curve.group)

        # -------------------------------------------------
        # Sub-socket visibility
        # -------------------------------------------------

        for control in self.sub_socket_control:
            cmds.connectAttr(
                f"{self.main_ctrl}.sub_socket",
                f"{control.transform}.visibility",
                force=True,
            )

            cmds.addAttr(
                control.transform,
                longName="sub_socket",
                proxy=f"{self.main_ctrl}.sub_socket",
            )

        # -------------------------------------------------
        # Selection set
        # -------------------------------------------------

        self.sub_socket_set = cmds.sets(
            [control.transform for control in self.sub_socket_control],  # type:ignore
            name=f"socket_sub_{self.side}_set",
        )
