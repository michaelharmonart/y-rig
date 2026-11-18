from maya import cmds
from maya.api.OpenMaya import MFnNurbsCurve, MMatrix, MSelectionList, MSpace, MVector

from yrig.control import Control, create_control
from yrig.joint import create_joint
from yrig.skin.split.tag import tag_for_weight_split
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.spline.matrix_spline.core import MatrixSpline
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
        divisions: int = 11,
    ) -> None:
        if divisions < 2:
            raise ValueError("Socket divisions must be at least 2")

        self.side = side
        self.guides = guides or {}
        self.main_ctrl = main_ctrl
        self.parent = parent
        self.joint_parent = joint_parent
        self.component_grp = component_grp
        self.control_grp = control_grp
        self.control_size = control_size
        self.divisions = divisions

    def disconnect_socket_joint_scale_shear(self) -> None:
        """Remove scale/shear drivers from socket joints."""

        for joints in self.sub_socket_joints.values():
            for joint in joints:
                for attr in (
                    "scale",
                    "scaleX",
                    "scaleY",
                    "scaleZ",
                    "shear",
                    "shearXY",
                    "shearXZ",
                    "shearYZ",
                ):
                    destination = f"{joint}.{attr}"

                    source = cmds.connectionInfo(
                        destination,
                        sourceFromDestination=True,
                    )

                    if source:
                        cmds.disconnectAttr(source, destination)  # type:ignore

                cmds.setAttr(
                    f"{joint}.scale",
                    1,  # type:ignore
                    1,  # type:ignore
                    1,  # type:ignore
                    type="double3",
                )

                cmds.setAttr(
                    f"{joint}.shear",
                    0,  # type:ignore
                    0,  # type:ignore
                    0,  # type:ignore
                    type="double3",
                )

    def create_eyelid_driver_spline(
        self,
        name: str,
        curve: str,
        inner_corner: str,
        middle_control: str,
        outer_corner: str,
        sub_controls: list[str],
        parent: str = "setup",
    ) -> MatrixSpline:
        """
        Build an eyelid matrix spline using five CV drivers.

        CV order:
            inner corner
            20% driver
            middle control
            80% driver
            outer corner

        The 20% and 80% drivers are parented under the middle
        control while preserving their initial world positions.
        """

        drivers = []

        for percent in (0.2, 0.8):
            driver = create_transform(
                name=f"{name}_{int(percent * 100)}_driver",
                parent=middle_control,
            )

            # Evaluate position along the source curve.
            shape = cmds.listRelatives(
                curve,
                shapes=True,
                noIntermediate=True,
                fullPath=True,
            )[0]

            min_u = cmds.getAttr(f"{shape}.minValue")
            max_u = cmds.getAttr(f"{shape}.maxValue")

            parameter = min_u + (max_u - min_u) * percent

            position = cmds.pointOnCurve(
                curve,
                parameter=parameter,
                position=True,
            )

            cmds.xform(
                driver,
                worldSpace=True,
                translation=tuple(position),  # type:ignore
            )

            drivers.append(driver)

        cv_transforms = [
            inner_corner,
            drivers[0],
            middle_control,
            drivers[1],
            outer_corner,
        ]

        spline: MatrixSpline = matrix_spline_from_transforms(
            name=f"{name}_spline",
            cv_transforms=cv_transforms,
            pinned_transforms=sub_controls,
            parent=parent,
            degree=3,
            padded=False,
            stretch=False,
            arc_length=True,
            align_tangent=True,
            interpolate_rotation=True,
            interpolate_scale=False,
        )

        return spline

    def get_curve_point_matrix(
        self,
        curve: str,
        parameter: float,
        tangent_orient: bool = True,
    ) -> MMatrix:
        """Sample a guide curve at a normalized *arc-length* fraction.

        Match the eyelid's tangent-based X-axis orientation. The world-space
        sample is static at build time; no live spline connections are made.
        """
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
        fn_curve = MFnNurbsCurve(selection.getDagPath(0))

        fraction = max(0.0, min(1.0, parameter))
        curve_parameter = fn_curve.findParamFromLength(fn_curve.length() * fraction)
        position = fn_curve.getPointAtParam(curve_parameter, MSpace.kWorld)

        if tangent_orient:
            aim = fn_curve.tangent(curve_parameter, MSpace.kWorld).normal()
            up = MVector(0, 1, 0)
            if abs(aim * up) > 0.999:
                up = MVector(0, 0, 1) if abs(aim.z) < 0.999 else MVector(1, 0, 0)
            z_axis = (aim ^ up).normal()
            y_axis = (z_axis ^ aim).normal()
            x_axis = aim
        else:
            x_axis = MVector(1, 0, 0)
            y_axis = MVector(0, 1, 0)
            z_axis = MVector(0, 0, 1)

        return MMatrix(
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

    def create_socket_follow(self, default_mult: float = 0.1) -> str:
        """Inherit control_grp once, and blend extra main-eye motion."""
        driver = self.main_ctrl

        if not cmds.attributeQuery("socket_follow", node=driver, exists=True):
            cmds.addAttr(
                driver,
                longName="socket_follow",
                attributeType="double",
                minValue=0.0,
                maxValue=1.0,
                defaultValue=default_mult,
                keyable=True,
            )

        follow_pos = create_transform(
            name=f"socket_{self.side}_follow_pos",
            parent=self.control_grp,
            transform=driver,
        )
        follow_offset = create_transform(
            name=f"socket_{self.side}_follow_offset",
            parent=follow_pos,
            transform=driver,
        )
        cmds.setAttr(f"{follow_offset}.translate", 0, 0, 0, type="double3")  # type:ignore
        cmds.setAttr(f"{follow_offset}.rotate", 0, 0, 0, type="double3")  # type:ignore

        controls_grp = create_transform(
            name=f"socket_{self.side}_controls_grp",
            parent=follow_offset,
        )

        relative_mm = cmds.createNode("multMatrix", name=f"socket_{self.side}_follow_relative_mm")
        cmds.connectAttr(f"{driver}.worldMatrix[0]", f"{relative_mm}.matrixIn[0]")
        cmds.connectAttr(
            f"{self.control_grp}.worldInverseMatrix[0]",
            f"{relative_mm}.matrixIn[1]",
        )

        rest_relative = MMatrix(cmds.getAttr(f"{relative_mm}.matrixSum"))
        delta_mm = cmds.createNode("multMatrix", name=f"socket_{self.side}_follow_delta_mm")
        cmds.setAttr(
            f"{delta_mm}.matrixIn[0]",
            *tuple(rest_relative.inverse()),
            type="matrix",
        )
        cmds.connectAttr(f"{relative_mm}.matrixSum", f"{delta_mm}.matrixIn[1]")

        blend = cmds.createNode("blendMatrix", name=f"socket_{self.side}_follow_bm")
        cmds.connectAttr(f"{delta_mm}.matrixSum", f"{blend}.target[0].targetMatrix")
        cmds.connectAttr(f"{driver}.socket_follow", f"{blend}.target[0].weight")

        local_mm = cmds.createNode("multMatrix", name=f"socket_{self.side}_follow_local_mm")
        cmds.connectAttr(f"{follow_pos}.matrix", f"{local_mm}.matrixIn[0]")
        cmds.connectAttr(f"{blend}.outputMatrix", f"{local_mm}.matrixIn[1]")
        cmds.connectAttr(f"{follow_pos}.inverseMatrix", f"{local_mm}.matrixIn[2]")
        cmds.connectAttr(f"{local_mm}.matrixSum", f"{follow_offset}.offsetParentMatrix")

        self.socket_follow_pos = follow_pos
        self.socket_follow_offset = follow_offset
        self.socket_follow_blend = blend
        self.socket_controls_grp = controls_grp
        return controls_grp

    def build_socket(self) -> None:
        """Build socket drivers and evenly spaced sub-controls, without a spline."""
        self.major_controls: dict[str, Control] = {}
        self.parent_controls: dict[str, Control] = {}
        self.corner_controls: dict[str, Control] = {}
        self.main_joints: dict[str, str] = {}
        self.sub_socket_control: list[Control] = []
        self.sub_socket_controls: dict[str, list[Control]] = {"upper": [], "lower": []}
        self.sub_socket_joints: dict[str, list[str]] = {"upper": [], "lower": []}
        self.socket_driver_controls: list[Control] = []

        self.socket_follow_grp = self.create_socket_follow(default_mult=0.1)

        # Every upper/lower sub-control is a direct child of this group.
        self.sub_socket_controls_grp = create_transform(
            name=f"sub_socket_controls_{self.side}_grp",
            parent=self.socket_follow_grp,
        )

        # Seven-point guides are retained ONLY to place the four main drivers.
        curve_guides = {}
        for vertical in ("upper", "lower"):
            curve_guides[vertical] = GuideCurve(
                curve=self.guides[f"socket_{vertical}_curve"],
                resample_amount=7,
                output_names=[
                    f"{vertical}_inner_corner",
                    f"{vertical}_inner_01",
                    f"{vertical}_inner_02",
                    f"{vertical}_mid",
                    f"{vertical}_outer_02",
                    f"{vertical}_outer_01",
                    f"{vertical}_outer_corner",
                ],
                ignore_handles=True,
                align_normals=True,
                mirror=self.side == "R",
            )

        # Main drivers are independent of the sub-controls.
        for corner, index in (("inner", 0), ("outer", -1)):
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

        for vertical in ("upper", "lower"):
            control = create_control(
                name=f"socket_{vertical}_{self.side}",
                parent=self.socket_follow_grp,
                transform=self.guides[f"socket_mid_{vertical}"],
                size=self.control_size / 2,
                control_shape="round_square",
                direction="z",
                dimensions=(1, 0.2, 0.2),
            )
            self.parent_controls[f"{vertical}_ctrl"] = control
            self.socket_driver_controls.append(control)
            cmds.addAttr(
                control.transform,
                longName="sub_socket",
                proxy=f"{self.main_ctrl}.sub_socket",
            )

        # Build evenly spaced sub-controls and their separate skeleton joints.
        for vertical in ("upper", "lower"):
            curve = self.guides[f"socket_{vertical}_curve"]
            joints: list[str] = []

            for index in range(self.divisions):
                percent = index / (self.divisions - 1)
                matrix = self.get_curve_point_matrix(
                    curve=curve,
                    parameter=percent,
                    tangent_orient=True,
                )
                name = f"socket_{vertical}_{index:02d}_{self.side}"

                control = create_control(
                    name=name,
                    parent=self.sub_socket_controls_grp,
                    transform=matrix,
                    size=self.control_size / 8,
                    control_shape="circle",
                    direction="z",
                )
                joint = create_joint(
                    name=name,
                    transform=control.transform,
                    parent=self.joint_parent,
                )
                self.major_controls[f"{name}_ctrl"] = control
                self.sub_socket_control.append(control)
                self.sub_socket_controls[vertical].append(control)
                joints.append(joint)

            # Preserve the old inner-corner joint name ONLY for the lead.
            lead_name = f"{vertical}_inner_corner_{self.side}_jnt"
            lead_joint = cmds.rename(joints[0], lead_name)
            joints[0] = lead_joint
            self.sub_socket_joints[vertical] = joints

            for index, joint in enumerate(joints):
                key = lead_name if index == 0 else f"socket_{vertical}_{index:02d}_{self.side}_jnt"
                self.main_joints[key] = joint

            tag_for_weight_split(
                influence=lead_joint,
                split_influences=joints,
            )

        for guide_curve in curve_guides.values():
            cmds.delete(guide_curve.group)

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

        self.sub_socket_set = cmds.sets(
            [control.transform for control in self.sub_socket_control],  # type:ignore
            name=f"socket_sub_{self.side}_set",
        )

        # ---------------------------------------------------------
        # Build upper and lower eyelid matrix splines
        # ---------------------------------------------------------

        self.socket_splines = {}

        for vertical in ("upper", "lower"):
            self.socket_splines[vertical] = self.create_eyelid_driver_spline(
                name=f"socket_{vertical}_{self.side}",
                curve=self.guides[f"socket_{vertical}_curve"],
                inner_corner=self.corner_controls["inner"].transform,
                middle_control=self.parent_controls[f"{vertical}_ctrl"].transform,
                outer_corner=self.corner_controls["outer"].transform,
                sub_controls=[control.transform for control in self.sub_socket_controls[vertical]],
                parent="setup",
            )
        # Prevent inverse global-scale compensation on socket joints.
        self.disconnect_socket_joint_scale_shear()
