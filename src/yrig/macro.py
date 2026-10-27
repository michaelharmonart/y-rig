import math
from collections.abc import Mapping

from maya.api.OpenMaya import MDGModifier, MFn, MFnDependencyNode, MFnUnitAttribute, MObject
from maya.api.OpenMayaAnim import MFnAnimCurve

from yrig.maya_api.attribute import Attribute
from yrig.maya_api.enum import KeyframeInfinityType, KeyframeTangentType
from yrig.maya_api.utils import get_plug


def set_driven_key(
    driver: Attribute | str,
    driven: Attribute | str,
    keys: Mapping[float, float],
    *,
    in_tangent: KeyframeTangentType = KeyframeTangentType.LINEAR,
    out_tangent: KeyframeTangentType = KeyframeTangentType.LINEAR,
    first_in_tangent: KeyframeTangentType | None = KeyframeTangentType.CLAMPED,
    first_out_tangent: KeyframeTangentType | None = None,
    last_in_tangent: KeyframeTangentType | None = None,
    last_out_tangent: KeyframeTangentType | None = KeyframeTangentType.CLAMPED,
    pre_infinity: KeyframeInfinityType = KeyframeInfinityType.LINEAR,
    post_infinity: KeyframeInfinityType = KeyframeInfinityType.LINEAR,
) -> str:
    driver_plug = get_plug(driver)
    driven_plug = get_plug(driven)

    curve_fn: MFnAnimCurve = MFnAnimCurve()
    curve_type = curve_fn.unitlessAnimCurveTypeForPlug(driven_plug)
    curve: MObject = curve_fn.create(driven_plug, curve_type)
    curve_nope: MFnDependencyNode = MFnDependencyNode(curve)

    modifier = MDGModifier()
    modifier.connect(driver_plug, curve_fn.findPlug("input", False))
    modifier.doIt()

    is_angle = driven_plug.attribute().hasFn(MFn.kUnitAttribute) and (
        MFnUnitAttribute(driven_plug.attribute()).unitType() == MFnUnitAttribute.kAngle
    )

    ordered_keys = sorted(keys.items())
    for driver_value, driven_value in ordered_keys:
        if is_angle:
            driven_value = math.radians(driven_value)
        key_in_tangent = in_tangent
        key_out_tangent = out_tangent
        curve_fn.addKey(
            driver_value,
            driven_value,
            key_in_tangent.ktype,
            key_out_tangent.ktype,
        )
    if ordered_keys:
        curve_fn.setInTangentType(
            0,
            (first_in_tangent or in_tangent).ktype,
        )
        curve_fn.setOutTangentType(
            0,
            (first_out_tangent or out_tangent).ktype,
        )

        last_index = curve_fn.numKeys - 1
        curve_fn.setInTangentType(
            last_index,
            (last_in_tangent or in_tangent).ktype,
        )
        curve_fn.setOutTangentType(
            last_index,
            (last_out_tangent or out_tangent).ktype,
        )
    curve_fn.setPreInfinityType(pre_infinity.ktype)
    curve_fn.setPostInfinityType(post_infinity.ktype)
    return curve_nope.name()
