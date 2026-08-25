from maya import cmds
from maya.api.OpenMaya import MMatrix
from mgear import pymaya as pm
from mgear.core import applyop, attribute, primitive, transform

# These are all values we assume, but that are properties of the mGear rig object and which we need to set.
RIG_ROOT_NODE: str = "rig"
ROTATION_OFFSET: tuple[float, float, float] = (0, 0, 0)
SEPARATE_JOINT_STRUCTURE: bool = True
USE_SEGMENT_SCALE_COMPENSATE: bool = False
ROOT_JOINT_PARENT: str = "jnt_org"
JOINT_WORLD_ORIENT: bool = False
FORCE_UNI_SCALE: bool = False
JOINTS_SET_NAME: str = f"{RIG_ROOT_NODE}_deformers_grp"
JOINT_VIS_ATTR: str = f"{RIG_ROOT_NODE}.jnt_vis"


def add_to_joint_set(joint: str) -> None:
    if cmds.objExists(JOINTS_SET_NAME) and cmds.nodeType(JOINTS_SET_NAME) == "objectSet":
        cmds.sets(joint, addElement=JOINTS_SET_NAME)


# This is taken from the mGear addJoint function and cleaned up since otherwise the interface is gross.
def add_joint(  # noqa: ANN201
    obj: str | MMatrix,
    name: str,
    parent: str | None,
    uni_scale: bool = False,
    seg_comp: bool = False,
    rot_off: tuple[float, float, float] | None = None,
    leaf_joint: bool = False,
    data_contracts=None,  # noqa: ANN001
    preBind_relative=None,  # noqa: ANN001, N803
    neutral_rot=True,  # noqa: ANN001
):
    """Add joint as child of the active joint or under driver object.

    This method uses the matrix contraint mgear_solver. If vanilla_nodes is
    set True, it will be bypass and sue the old method

    Args:
        obj (dagNode): The input driver object for the joint.
        name (str): The joint name.
        newActiveJnt (bool or dagNode): If a joint is pass, this joint will
            be the active joint and parent of the newly created joint.
        UniScale (bool): Connects the joint scale with the Z axis for a
            unifor scalin, if set Falsewill connect with each axis
            separated.
        segComp (bool): Set True or False the segment compensation in the
            joint..
        rot_off (list, optional): offset in degrees for XYZ rotation
        leaf_joint (bool, optional): If true will create a child joint as
            a leaf joint to  imput the scale. This option is meant for games
        guide_relative (str, optional): Guide locator name tha define joint
            position
        preBind_relative (dagNode): if the argument is set will create a
            message connection to track the prebind reference of the joint
            to use in the skinning prebind matrix (like Doritos technique ;) .


    Deleted Parameters:
        dagNode: The newly created joint.

    """

    # force SSC override
    if USE_SEGMENT_SCALE_COMPENSATE:
        seg_comp = True

    if not rot_off:
        rot_off = (0, 0, 0)
    custom_name = name

    if SEPARATE_JOINT_STRUCTURE:
        rule_name = name
        jnt = name

        if isinstance(obj, MMatrix):
            t = obj
        else:
            t = transform.getTransform(obj)
        parent = ROOT_JOINT_PARENT if parent is None else parent
        parent_node = pm.PyNode(parent)
        jnt = primitive.addJoint(parent_node, custom_name or rule_name, t)
        keep_off = False

        # check if already have connections
        # for example Mehahuman twist joint already have connections
        if not attribute.has_in_connections(jnt):
            # Disconnect inversScale for better preformance
            if isinstance(parent_node, pm.nodetypes.Joint):
                try:
                    pm.disconnectAttr(parent_node.scale, jnt.inverseScale)

                except RuntimeError:
                    # This handle the situation where we have in between
                    # joints transformation due a negative scaling
                    if not isinstance(jnt, pm.nodetypes.Joint):
                        pm.ungroup(jnt.getParent())  # type: ignore

            if keep_off:
                # if jnt.rotate.get() != (0, 0, 0):
                if not all(component == 0 for component in jnt.rotate.get()):
                    pm.displayInfo(
                        f"Joint {jnt.name()} has non-zero rotations, We will use Constraints to connect: {jnt.rotate.get()}"
                    )
                if isinstance(obj, MMatrix):
                    driver = None
                    jnt.setMatrix(obj, worldSpace=True)
                else:
                    driver = primitive.addTransform(obj, name=obj.name() + "_cnx_off")
                    transform.matchWorldTransform(jnt, driver)
                    rot_off = [0, 0, 0]

            else:
                if isinstance(obj, MMatrix):
                    driver = None
                    jnt.setMatrix(obj, worldSpace=True)

                else:
                    driver = obj
                    rot_off = rot_off  # noqa

            if driver:
                if JOINT_WORLD_ORIENT:
                    driver = primitive.addTransformFromPos(
                        driver,
                        name=obj + "_world_ori",
                        pos=transform.getTranslation(driver),
                    )

                cns_m = applyop.gear_matrix_cns(driver, jnt, rot_off=rot_off, connect_srt="srt")

                # invert negative scaling in Joints. We only inver Z axis,
                # so is the only axis that we are checking
                if jnt.scaleZ.get() < 0:
                    cns_m.scaleMultZ.set(-1.0)
                    cns_m.rotationMultX.set(-1.0)
                    cns_m.rotationMultY.set(-1.0)

                # if unifor scale is False by default. It can be forced
                # using uniScale arg or set from the ui
                if FORCE_UNI_SCALE:
                    uni_scale = True
                if uni_scale:
                    attribute.disconnect_inputs(jnt, ["scale"])
                    pm.connectAttr(cns_m.scaleZ, jnt.sx)  # type: ignore
                    pm.connectAttr(cns_m.scaleZ, jnt.sy)  # type: ignore
                    pm.connectAttr(cns_m.scaleZ, jnt.sz)  # type: ignore

                # leaf joint
                if leaf_joint and not uni_scale:
                    leaf_jnt = primitive.addJoint(jnt, "leaf_" + jnt.name(), t)
                    leaf_jnt.attr("radius").set(1.5)
                    leaf_jnt.attr("overrideEnabled").set(1)
                    leaf_jnt.attr("overrideColor").set(13)
                    leaf_jnt.rotate.set([0, 0, 0])
                    # create and connect message to track the leaf joint relation
                    if not jnt.hasAttr("leaf_joint"):
                        pm.addAttr(jnt, ln="leaf_joint", at="message", m=True)  # type: ignore
                    pm.connectAttr(leaf_jnt.message, jnt.leaf_joint)  # type: ignore

                    add_to_joint_set(str(leaf_jnt))
                    # connect scale
                    jnt.disconnectAttr("scale")
                    jnt.disconnectAttr("shear")
                    pm.connectAttr(cns_m.scale, leaf_jnt.scale)  # type: ignore
                    pm.connectAttr(cns_m.shear, leaf_jnt.shear)  # type: ignore

                if preBind_relative:
                    pm.addAttr(jnt, ln="preBind_relative", at="message", m=False)  # type: ignore
                    pm.connectAttr(preBind_relative.message, jnt.preBind_relative)  # type: ignore

            else:
                cns_m = None

            # Segment scale compensate on/Off
            # TODO: before was always off to avoid issues with the
            # global scale. Confirm there is no conflicts
            jnt.setAttr("segmentScaleCompensate", seg_comp)

            if not keep_off and neutral_rot:
                # setting the joint orient compensation in order to
                # have clean rotation channels
                jnt.setAttr("jointOrient", 0, 0, 0)
                if cns_m:
                    m = cns_m.drivenRestMatrix.get()
                else:
                    driven_m = pm.getAttr(jnt + ".parentInverseMatrix[0]")
                    m = t * driven_m
                    jnt.rotate.set([0, 0, 0])
                    if jnt.scaleZ.get() < 0:
                        jnt.scaleZ.set(1)
                tm = pm.datatypes.TransformationMatrix(m)
                r = pm.datatypes.degrees(tm.getRotation())
                jnt.jointOrient.set([r[0], r[1], r[2]])
            elif not neutral_rot:
                jnt.setAttr("jointOrient", 0, 0, 0)
                driven_m = pm.getAttr(jnt + ".parentInverseMatrix[0]")
                m = t * driven_m
                tm = pm.datatypes.TransformationMatrix(m)
                r = pm.datatypes.degrees(tm.getRotation())
                jnt.rotate.set([r[0], r[1], r[2]])

            # set not keyable
            attribute.setNotKeyableAttributes(jnt)

    else:
        jnt = primitive.addJoint(
            obj,
            custom_name,
            transform.getTransform(obj),
        )
        pm.connectAttr(JOINT_VIS_ATTR, jnt.attr("visibility"))  # type: ignore
        attribute.lockAttribute(jnt)
    add_to_joint_set(str(jnt))

    if data_contracts:
        if not jnt.hasAttr("data_contracts"):
            attribute.addAttribute(jnt, "data_contracts", "string")
        jnt.data_contracts.set(data_contracts)

    return jnt
