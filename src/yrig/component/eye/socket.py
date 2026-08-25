import math

from maya import cmds
from maya.api.OpenMaya import MEulerRotation, MMatrix, MSpace, MTransformationMatrix, MVector

from yrig.control import create_control
from yrig.joint import create_joint
from yrig.skin.split.tag import tag_for_weight_split
from yrig.spline.matrix_spline.build import matrix_spline_from_transforms
from yrig.transform import create_transform
from yrig.transform.utils import get_position

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

        # Ensure we are working with the shape node
        shapes = cmds.listRelatives(curve, shapes=True, fullPath=True) or []
        if shapes:
            working_curve = shapes[0]

        top_grp = create_transform(name=f"{descriptor}_spline_{self.side}_grp", parent=parent)

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

    def build_socket(self) -> None:
        self.major_controls = {}
        self.parent_controls = {}
        self.main_joints = {}
        self.sub_socket_control = []
        major_guides = [
            "socket_inner_upper",
            "socket_mid_upper",
            "socket_outer_upper",
            "socket_inner_lower",
            "socket_mid_lower",
            "socket_outer_lower",
            "socket_inner_corner",
            "socket_outer_corner",
        ]

        for side in ["upper", "lower"]:
            self.parent_controls[f"{side}_ctrl"] = create_control(
                name=f"socket_{side}_{self.side}",
                parent=self.main_ctrl,
                transform=self.guides[f"socket_mid_{side}"],
                size=self.control_size / 2,
                control_shape="round_square",
                direction="z",
                dimensions=(1, 0.2, 0.2),
            )

            cmds.addAttr(
                self.parent_controls[f"{side}_ctrl"].transform,
                longName="sub_socket",
                proxy=f"{self.main_ctrl}.sub_socket",
            )

        for x, side in enumerate(["upper", "lower"]):
            curveguides = GuideCurve(
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
            )
            jnt_list = []
            for i, guide in enumerate(curveguides.locator_list):
                if x == 0 and i in [1, 2, 3, 4, 5]:
                    parent = self.parent_controls["upper_ctrl"]
                elif x == 1 and i in [1, 2, 3, 4, 5]:
                    parent = self.parent_controls["lower_ctrl"]
                else:
                    parent = self.main_ctrl
                self.major_controls[f"{guide.name}_ctrl"] = create_control(
                    name=f"{guide.name}_{self.side}",
                    parent=parent,
                    transform=guide.name,
                    size=self.control_size / 8,
                    control_shape="circle",
                    direction="z",
                )

                self.sub_socket_control.append(self.major_controls[f"{guide.name}_ctrl"])
                self.main_joints[f"{guide.name}_jnt"] = create_joint(
                    name=f"{guide.name}_{self.side}",
                    transform=self.major_controls[f"{guide.name}_ctrl"].transform,
                    parent=self.joint_parent,
                )

                jnt_list.append(self.main_joints[f"{guide.name}_jnt"])
            tag_for_weight_split(
                influence=jnt_list[0],  # <-- your SOURCE joint (must already exist)
                split_influences=jnt_list,  # <-- the ones you just created
            )
            cmds.delete(curveguides.group)

        for control in self.sub_socket_control:
            cmds.connectAttr(f"{self.main_ctrl}.sub_socket", f"{control.transform}.visibility")
            cmds.addAttr(
                f"{control.transform}",
                longName="sub_socket",
                proxy=f"{self.main_ctrl}.sub_socket",
            )
