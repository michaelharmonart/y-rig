from __future__ import annotations

from dataclasses import dataclass, field

from maya import cmds
from maya.api.OpenMaya import MMatrix

from yrig.joint import create_joint
from yrig.transform.matrix import set_local_matrix
from yrig.transform.utils import create_transform, match_transform

from .curve import create_line_curve, curve_cvs, style_curve

GUIDE_TAG = "isGuide_TAG"


@dataclass
class GuideInfo:
    name: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float]
    guide_type: str

    extra_channels: list[str] = field(default_factory=list)
    descriptor: str = ""
    side: str = ""

    modules: list = field(default_factory=list)


@dataclass
class SplineGuideInfo:
    curve: GuideInfo
    lower: GuideInfo
    lower_tan: GuideInfo
    upper: GuideInfo
    upper_tan: GuideInfo


@dataclass
class ChainGuideInfo:
    guides: list[GuideInfo]


def mirror_guide(guide: GuideInfo) -> GuideInfo | None:
    if guide.guide_type == "joint":
        temp_parent = create_transform(name="temp_grp")
        parent = cmds.listRelatives(guide.name, parent=True)

        new_guide = create_guide_from_position(
            pos=guide.pos,
            guide_name=guide.descriptor.replace("_L", "_R"),
            parent=temp_parent,
            component_type="joint",  # type:ignore
        )

        cmds.setAttr(f"{new_guide.name}.rotate", *guide.rot)  # type:ignore
        cmds.setAttr(f"{temp_parent}.scaleX", -1)  # type:ignore

        cmds.parent(new_guide.name, parent[0])

        cmds.delete(temp_parent)

        return new_guide
    return None


def align_guides(
    guide_01: GuideInfo,
    guide_02: GuideInfo,
    primary_axis: str = "X",
    flip: bool = False,
) -> GuideInfo:

    axis = primary_axis.upper()

    axis_vectors = {
        "X": (1, 0, 0),
        "Y": (0, 1, 0),
        "Z": (0, 0, 1),
    }

    if axis not in axis_vectors:
        raise ValueError(f"Invalid primary_axis: {primary_axis}. Expected X, Y, or Z.")

    aim_vector = axis_vectors[axis]

    if flip:
        aim_vector = tuple(-value for value in aim_vector)

    # Pick an up axis that is not the primary axis
    up_vectors = {
        "X": (0, 1, 0),
        "Y": (0, 0, 1),
        "Z": (0, 1, 0),
    }

    up_vector = up_vectors[axis]

    aim = cmds.aimConstraint(
        guide_02.name,
        guide_01.name,
        aimVector=aim_vector,  # type:ignore
        upVector=up_vector,
        worldUpType="vector",
        worldUpVector=up_vector,
        maintainOffset=False,
    )

    cmds.delete(aim)  # type:ignore

    rotation = cmds.xform(
        guide_01.name,
        query=True,
        rotation=True,
        worldSpace=True,
    )

    guide_01.rot = (  # type:ignore
        rotation[0],  # type:ignore
        rotation[1],  # type:ignore
        rotation[2],  # type:ignore
    )

    return guide_01


def create_guide_from_position(
    pos: str | list | tuple | MMatrix | None,
    guide_name: str,
    parent: str | None = None,
) -> GuideInfo:

    guide = create_joint(
        name=f"{guide_name}_guide",
        connect=False,
        parent=parent,
        suffix=False,  # type:ignore
    )

    if isinstance(pos, str):
        if not cmds.objExists(pos):
            # print(f"{pos} does not exist")
            return None  # type:ignore

        match_transform(
            transform=guide,
            target_transform=pos,
        )

    elif isinstance(pos, (list, tuple)):
        cmds.xform(
            guide,
            worldSpace=True,
            translation=pos,  # type:ignore
        )

    elif isinstance(pos, MMatrix):
        set_local_matrix(
            transform=guide,
            matrix=pos,
            use_joint_orient=False,
        )

    return_pos = cmds.xform(
        guide,
        query=True,
        translation=True,
        worldSpace=True,
    )

    return_rot = cmds.xform(
        guide,
        query=True,
        rotation=True,
        worldSpace=True,
    )

    add_guide_metadata(
        guide=guide,
        descriptor=guide_name,
        guide_type="joint",
    )

    set_guide_tag(guide)

    return GuideInfo(
        name=guide,
        pos=tuple(return_pos),  # type:ignore
        rot=tuple(return_rot),  # type:ignore
        guide_type="joint",
        descriptor=guide_name,
    )


def read_guide(guide: str) -> GuideInfo:
    """Create GuideInfo from an existing Maya guide."""

    if not cmds.objExists(guide):
        raise ValueError(f"Guide does not exist: {guide}")

    pos = cmds.xform(
        guide,
        query=True,
        worldSpace=True,
        translation=True,
    )

    rot = cmds.xform(
        guide,
        query=True,
        worldSpace=True,
        rotation=True,
    )

    # -------------------
    # Guide type
    # -------------------

    if cmds.attributeQuery(
        "guidetype_TAG",
        node=guide,
        exists=True,
    ):
        guide_type = cmds.getAttr(f"{guide}.guidetype_TAG")

    else:
        node_type = cmds.nodeType(guide)

        if node_type == "joint":
            guide_type = "joint"

        elif node_type == "transform":
            shapes = (
                cmds.listRelatives(
                    guide,
                    shapes=True,
                )
                or []
            )

            if shapes and cmds.nodeType(shapes[0]) == "nurbsCurve":
                guide_type = "curve"
            else:
                guide_type = "transform"

        else:
            guide_type = node_type

        add_guide_metadata(
            guide=guide,
            descriptor=guide.removesuffix("_guide"),
            guide_type=guide_type,  # type:ignore
        )

    # -------------------
    # Descriptor
    # -------------------

    descriptor = guide.removesuffix("_guide")

    # -------------------
    # Extra channels
    # -------------------

    extra_channels = []

    if cmds.attributeQuery(
        "guideExtraChannels",
        node=guide,
        exists=True,
    ):
        channel_string = cmds.getAttr(f"{guide}.guideExtraChannels") or ""

        extra_channels = [channel for channel in channel_string.split(",") if channel]

    # -------------------
    # Side
    # -------------------

    if "_L_" in guide:
        side = "L"
    elif "_R_" in guide:
        side = "R"
    elif "_M_" in guide:
        side = "M"
    else:
        side = ""

    return GuideInfo(
        name=guide,
        pos=tuple(pos),  # type:ignore
        rot=tuple(rot),  # type:ignore
        guide_type=guide_type,  # type:ignore
        extra_channels=extra_channels,
        descriptor=descriptor,
        side=side,
    )


def add_guide_metadata(
    guide: str,
    descriptor: str,
    guide_type: str,
) -> None:
    """Store guide identification data directly on a Maya node."""

    attrs = {
        "descriptor_TAG": descriptor,
        "guidetype_TAG": guide_type,
    }

    for attr, value in attrs.items():
        if not cmds.attributeQuery(attr, node=guide, exists=True):
            cmds.addAttr(
                guide,
                longName=attr,
                dataType="string",
            )

        cmds.setAttr(
            f"{guide}.{attr}",
            value,
            type="string",
        )


def create_spline_guide(
    parent: str,
    lower_name: str = "lower",
    upper_name: str = "upper",
    curve_name: str = "spline",
    side: str = "M",
    position: list | None = None,
) -> SplineGuideInfo:
    if position:
        upper_pos = position[0]
        lower_pos = position[1]
    else:
        upper_pos = (0, 5, 0)
        lower_pos = (0, 0, 0)

    upper = create_guide_from_position(
        pos=upper_pos, guide_name=f"{upper_name}_{side}", parent=parent
    )
    lower = create_guide_from_position(
        pos=lower_pos, guide_name=f"{lower_name}_{side}", parent=parent
    )

    curve = create_line_curve(
        name=f"{curve_name}_{side}_guide", start_position=lower_pos, end_position=upper_pos
    )
    style_curve(curve=curve, line_width=5, draw_on_top=True, template=True)
    cmds.displaySmoothness(
        curve,
        pointsWire=16,
        pointsShaded=4,
        polygonObject=3,
    )
    cmds.parent(curve, parent)

    curve_guide = GuideInfo(
        name=f"{curve_name}_{side}_guide",
        pos=(0, 0, 0),
        rot=(0, 0, 0),
        guide_type="curve",
        extra_channels=[],
        descriptor=f"{upper_name}_{side}",
    )

    temp = curve_cvs(curve=curve, clusters=True)
    pos = temp[1]  # type:ignore
    clusters = temp[0]

    upper_tan = create_guide_from_position(
        pos=tuple(pos[2]), guide_name=f"{upper_name}_tan_{side}", parent=upper.name
    )
    lower_tan = create_guide_from_position(
        pos=tuple(pos[1]), guide_name=f"{lower_name}_tan_{side}", parent=lower.name
    )

    guides = [lower, lower_tan, upper_tan, upper]

    for i, cluster in enumerate(clusters):
        # match_transform(transform=cluster, target_transform=guides[i].name)
        cmds.parent(cluster, guides[i].name)
        tform = cmds.listRelatives(parent=True)[0]
        if tform == guides[i].name:
            cmds.setAttr(f"{cluster}.visibility", False)  # type:ignore
            cmds.setAttr(f"{cluster}.hiddenInOutliner", True)  # type:ignore
        else:
            tform = cmds.rename(tform, f"{guides[i].name}_offset")
            cmds.setAttr(f"{tform}.visibility", False)  # type:ignore
            cmds.setAttr(f"{tform}.hiddenInOutliner", True)  # type:ignore

    info = SplineGuideInfo(
        curve=curve_guide, upper=upper, upper_tan=upper_tan, lower=lower, lower_tan=lower_tan
    )
    return info


def chain_guides(
    names: list, side: str = "M", position: list | None = None, parent: str = ""
) -> ChainGuideInfo:
    guide_list = []
    for i, g in enumerate(names):
        guide = create_guide_from_position(
            pos=position[i] if position else (0, i, 0), guide_name=f"{g}_{side}", parent=parent
        )
        guide_list.append(guide)
    info = ChainGuideInfo(guides=guide_list)
    return info


def create_midpoint_guide(
    guide_a: GuideInfo | str,
    guide_b: GuideInfo | str,
    name: str | None = None,
    parent: str | None = None,
    component_type: str | None = None,
    position_only: bool = True,
    create: bool = True,
) -> GuideInfo | tuple[float, float, float]:
    """
    Create a guide halfway between two existing guides.

    Args:
        guide_a: First guide or Maya node name.
        guide_b: Second guide or Maya node name.
        name: Optional name override.
        parent: Parent for the new guide.
        component_type: Component metadata.
        position_only: Ignore input rotations and scales.
            When False, inherit guide A's world rotation.
        create: If False, return the midpoint position
            without creating a guide.

    Returns:
        GuideInfo if create=True, otherwise midpoint position.
    """

    if isinstance(guide_a, str):
        guide_a = read_guide(guide_a)

    if isinstance(guide_b, str):
        guide_b = read_guide(guide_b)

    # Always query current scene positions.
    pos_a = cmds.xform(
        guide_a.name,
        query=True,
        worldSpace=True,
        translation=True,
    )

    pos_b = cmds.xform(
        guide_b.name,
        query=True,
        worldSpace=True,
        translation=True,
    )

    midpoint = tuple((a + b) * 0.5 for a, b in zip(pos_a, pos_b, strict=False))  # type:ignore

    # Return the coordinates without creating anything.
    if not create:
        return midpoint

    # Derive a name if none was supplied.
    if name is None:
        name_a = guide_a.descriptor or guide_a.name.removesuffix("_guide")
        name_b = guide_b.descriptor or guide_b.name.removesuffix("_guide")

        name = f"{name_a}_{name_b}_mid"

    # Default to guide A's parent.
    if parent is None:
        parents = (
            cmds.listRelatives(
                guide_a.name,
                parent=True,
                fullPath=True,
            )
            or []
        )

        parent = parents[0] if parents else None

    # Create the guide using the existing Workshop helper.
    midpoint_guide = create_guide_from_position(
        pos=midpoint,
        guide_name=name,
        parent=parent,
        component_type=component_type,  # type:ignore
    )

    if not position_only:
        rotation = cmds.xform(
            guide_a.name,
            query=True,
            worldSpace=True,
            rotation=True,
        )

        cmds.xform(
            midpoint_guide.name,
            worldSpace=True,
            rotation=rotation,  # type:ignore
        )

    else:
        cmds.xform(
            midpoint_guide.name,
            worldSpace=True,
            rotation=(0, 0, 0),
        )

    # Keep GuideInfo synchronized with the Maya node.
    midpoint_guide.pos = tuple(
        cmds.xform(
            midpoint_guide.name,
            query=True,
            worldSpace=True,
            translation=True,
        )  # type:ignore
    )

    midpoint_guide.rot = tuple(
        cmds.xform(
            midpoint_guide.name,
            query=True,
            worldSpace=True,
            rotation=True,
        )  # type:ignore
    )

    return midpoint_guide


def is_guide(node: str) -> bool:
    """
    Return whether a node is a Workshop guide.

    Supports:
        - Joint guides
        - Transform guides
        - Curve guides
    """

    if not cmds.objExists(node):
        return False

    # -----------------------------------------
    # EXPLICIT GUIDE TAG
    # -----------------------------------------

    if cmds.attributeQuery(
        GUIDE_TAG,
        node=node,
        exists=True,
    ):
        return bool(cmds.getAttr(f"{node}.{GUIDE_TAG}"))

    # -----------------------------------------
    # GUIDE TYPE METADATA
    # -----------------------------------------

    if cmds.attributeQuery(
        "guideType",
        node=node,
        exists=True,
    ):
        guide_type = cmds.getAttr(f"{node}.guideType")

        if guide_type in (
            "joint",
            "transform",
            "curve",
        ):
            return True

    return False


def set_guide_tag(
    node: str,
    state: bool = True,
) -> None:
    """
    Explicitly mark or unmark a node as a Workshop guide.
    """

    if not cmds.objExists(node):
        raise ValueError(f"Node does not exist: {node}")

    if not cmds.attributeQuery(
        GUIDE_TAG,
        node=node,
        exists=True,
    ):
        cmds.addAttr(
            node,
            longName=GUIDE_TAG,
            attributeType="bool",
            defaultValue=state,
        )

    cmds.setAttr(
        f"{node}.{GUIDE_TAG}",
        state,  # type:ignore
    )


def tag_existing_joint_guides(
    root: str = "root_M_guide",
) -> list[str]:
    """
    Add the explicit guide tag to existing joint guides.

    Intended as a migration utility for older guide scenes.
    """

    if not cmds.objExists(root):
        raise ValueError(f"Root does not exist: {root}")

    descendants = (
        cmds.listRelatives(
            root,
            allDescendents=True,
            fullPath=False,
        )
        or []
    )

    nodes = [root, *descendants]

    tagged = []

    for node in nodes:
        if cmds.nodeType(node) not in ("joint", "curve"):
            continue

        if not node.endswith("_guide"):
            continue

        set_guide_tag(node)
        tagged.append(node)

    return tagged
