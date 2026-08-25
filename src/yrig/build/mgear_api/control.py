from collections.abc import Callable

from maya import cmds
from maya.api.OpenMaya import MMatrix
from mgear import pymaya as pm
from mgear.core import attribute, icon, node
from mgear.core.attribute import setRotOrder

from yrig.maya_api import MAYA_API_VERSION
from yrig.maya_api.version import supports_shape_draw_on_top
from yrig.transform import set_world_matrix

# These are hardcoded despite the fact that any mGear guide can specify these.
# That's fine for now since we want this to be standard between our rigs anyways.
LEFT_SIDE_LABEL = "L"
RIGHT_SIDE_LABEL = "R"
MIDDLE_SIDE_LABEL = "M"
CONTROL_NAME_SUFFIX = "_ctl"
RIG_ROOT_NODE = "rig"
CONTROLS_SET_NAME = f"{RIG_ROOT_NODE}_controllers_grp"
CONTROL_XRAY_ATTR = f"{RIG_ROOT_NODE}.ctl_x_ray"


# This is taken from the mGear addCtl function and cleaned up since otherwise the interface is gross.
def add_ctl(  # noqa: ANN201
    name: str,
    parent: str | None,
    matrix: MMatrix | None,
    side: str,
    component: str | None = None,
    color: int | tuple[float, float, float] | None = None,
    icon_shape: str | None = None,
    tp: str | None = None,
    add_to_control_set: bool = True,
    control_icon_creator: Callable[[], str] | None = None,
    rotation_order: str | None = None,
    **kwargs,
):
    """
    Create the control and apply the shape, if this is alrealdy stored
    in the guide controllers grp.

    Args:
        parent (dagNode): The control parent
        name (str): The control name.
        m (matrix): The transfromation matrix for the control.
        color (int or list of float): The color for the control in index or
            RGB.
        iconShape (str): The controls default shape.
        tp (dagNode): Tag Parent Control object to connect as a parent
            controller
        kwargs (variant): Other arguments for the iconShape type variations

    Returns:
        dagNode: The Control.

    """
    if "degree" not in kwargs:
        kwargs["degree"] = 1

    if control_icon_creator is None:
        ctl = icon.create(parent, name, matrix, color, icon_shape, **kwargs)

    else:
        ctl = pm.PyNode(control_icon_creator())
        if parent is not None:
            pm.parent(ctl, parent, relative=True)  # type: ignore
        if matrix is not None:
            set_world_matrix(str(ctl), matrix)

    # add metadata attirbutes.
    attribute.addAttribute(ctl, "isCtl", "bool", keyable=False)
    attribute.addAttribute(ctl, "uiHost", "string", keyable=False)
    ctl.addAttr("uiHost_cnx", at="message", multi=False)

    role_name = name
    attribute.addAttribute(ctl, "ctl_role", "string", keyable=False, value=role_name)

    # mgear name. This keep track of the default shifter name. This naming
    # system ensure that each control has a unique id. Tools like mirror or
    # flip pose can use it to track symmetrical controls
    attribute.addAttribute(
        ctl,
        "shifter_name",
        "string",
        keyable=False,
        value=name,
    )
    attribute.addAttribute(ctl, "side_label", "string", keyable=False, value=side)
    attribute.addAttribute(
        ctl,
        "L_custom_side_label",
        "string",
        keyable=False,
        value=LEFT_SIDE_LABEL,
    )
    attribute.addAttribute(
        ctl,
        "R_custom_side_label",
        "string",
        keyable=False,
        value=RIGHT_SIDE_LABEL,
    )
    attribute.addAttribute(
        ctl,
        "C_custom_side_label",
        "string",
        keyable=False,
        value=MIDDLE_SIDE_LABEL,
    )

    attribute.addEnumAttribute(
        ctl,
        "rotate_order",
        0,
        ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx"),
        keyable=False,
    )

    # create the attributes to handle mirror and symmetrical pose
    attribute.add_mirror_config_channels(ctl)
    if (
        add_to_control_set
        and cmds.objExists(CONTROLS_SET_NAME)
        and cmds.nodeType(CONTROLS_SET_NAME) == "objectSet"
    ):
        cmds.sets(str(ctl), addElement=CONTROLS_SET_NAME)

    # Set the control shapes isHistoricallyInteresting
    # Use cmds for faster shape operations
    ctl_name = ctl.name()
    shapes = cmds.listRelatives(ctl_name, shapes=True, fullPath=True) or []
    maya_version = MAYA_API_VERSION
    for shape in shapes:
        cmds.setAttr(f"{shape}.isHistoricallyInteresting", False)  # type: ignore
        # connecting the always draw shapes on top to global attribute
        if supports_shape_draw_on_top() and cmds.objExists(CONTROL_XRAY_ATTR):
            cmds.connectAttr(CONTROL_XRAY_ATTR, f"{shape}.alwaysDrawOnTop")

    # set controller tag
    if maya_version >= 201650:
        try:
            oldTag = pm.PyNode(ctl.name() + "_tag")  # noqa:  N806
            if not oldTag.controllerObject.connections():
                # NOTE:  The next line is comment out. Because this will
                # happend alot since core does't clean
                # controller tags after deleting the control Object of the
                # tag. This have been log to Autodesk.
                # If orphane tags are found, it will be clean in silence.
                # pm.displayWarning(
                #     "Orphane Tag: %s  will be delete and created new for: %s"
                #     % (oldTag.name(), ctl.name())
                # )
                pm.delete(oldTag)
        except:  # noqa
            pass

        node.add_controller_tag(ctl, tp)

    # connect control message to component
    if component is not None:
        component_node = pm.PyNode(component)
        ni = attribute.get_next_available_index(component_node.compCtl)
        pm.connectAttr(ctl.message, component_node.attr(f"compCtl[{ni!s}]"))  # type: ignore

        ctl.addAttr("compRoot", at="message", m=False)
        component_node.message >> ctl.compRoot

    if rotation_order is not None:
        setRotOrder(ctl, rotation_order)

    return ctl
