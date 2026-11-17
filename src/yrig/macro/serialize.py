import logging

from maya import cmds
from maya.api.OpenMaya import MFnDependencyNode, MObject
from maya.api.OpenMayaAnim import MFnAnimCurve
from pxr import Sdf, Usd, Vt

from yrig.maya_api.enum import (
    KeyframeInfinityType,
    KeyframeTangentType,
)
from yrig.maya_api.node import AnimCurveUU
from yrig.maya_api.node.anim_curve import AnimCurveNode
from yrig.maya_api.utils import get_depend_node, get_plug

log = logging.getLogger(__name__)


def _create_attr(prim: Sdf.PrimSpec, name: str, type_name: Sdf.ValueTypeName) -> Sdf.AttributeSpec:
    attr_spec = Sdf.AttributeSpec(prim, name, type_name, Sdf.VariabilityUniform, False)
    return attr_spec


def get_scene_driven_keys() -> list[str]:
    return cmds.ls(type=["animCurveUA", "animCurveUL", "animCurveUT", "animCurveUU"]) or []


def define_prim_from_driven_key(
    layer: Sdf.Layer, path: Sdf.Path | str, driven_key: AnimCurveNode
) -> Sdf.PrimSpec:
    driven_key_prim = Sdf.CreatePrimInLayer(layer, path)
    driven_key_prim.specifier = Sdf.SpecifierDef
    driven_key_prim.typeName = "AnimCurve"

    input_attr = _create_attr(driven_key_prim, "input", Sdf.ValueTypeNames.String)
    input_attr.default = str(driven_key.input.get_input() or "")

    output_attr = _create_attr(driven_key_prim, "output", Sdf.ValueTypeNames.StringArray)
    output_attr.default = [str(out) for out in driven_key.output.get_outputs()]

    key_time_attr = _create_attr(driven_key_prim, "keyTime", Sdf.ValueTypeNames.DoubleArray)
    key_time_attr.default = [attr.key_time.get() for attr in driven_key.key_time_value]

    key_value_attr = _create_attr(driven_key_prim, "keyValue", Sdf.ValueTypeNames.DoubleArray)
    key_value_attr.default = [attr.key_value.get() for attr in driven_key.key_time_value]

    key_tan_in_attr = _create_attr(driven_key_prim, "keyTanIn", Sdf.ValueTypeNames.Double2Array)
    key_tan_in_attr.default = [
        (x.get(), y.get())
        for x, y in zip(driven_key.key_tan_in_x, driven_key.key_tan_in_y, strict=True)
    ]

    key_tan_in_type_attr = _create_attr(
        driven_key_prim, "keyTanInType", Sdf.ValueTypeNames.TokenArray
    )
    key_tan_in_type_attr.default = [token.get().usd_token for token in driven_key.key_tan_in_type]

    key_tan_out_attr = _create_attr(driven_key_prim, "keyTanOut", Sdf.ValueTypeNames.Double2Array)
    key_tan_out_attr.default = [
        (x.get(), y.get())
        for x, y in zip(driven_key.key_tan_out_x, driven_key.key_tan_out_y, strict=True)
    ]

    key_tan_out_type_attr = _create_attr(
        driven_key_prim, "keyTanOutType", Sdf.ValueTypeNames.TokenArray
    )
    key_tan_out_type_attr.default = [token.get().usd_token for token in driven_key.key_tan_out_type]

    key_tan_locked_attr = _create_attr(
        driven_key_prim, "keyTanLocked", Sdf.ValueTypeNames.BoolArray
    )
    key_tan_locked_attr.default = [attr.get() for attr in driven_key.key_tan_locked]

    key_weight_locked_attr = _create_attr(
        driven_key_prim, "keyWeightLocked", Sdf.ValueTypeNames.BoolArray
    )
    key_weight_locked_attr.default = [attr.get() for attr in driven_key.key_weight_locked]

    pre_inifinity_attr = _create_attr(driven_key_prim, "preInfinity", Sdf.ValueTypeNames.Token)
    pre_inifinity_attr.default = driven_key.pre_infinity.get().usd_token

    post_inifinity_attr = _create_attr(driven_key_prim, "postInfinity", Sdf.ValueTypeNames.Token)
    post_inifinity_attr.default = driven_key.post_infinity.get().usd_token

    weighted_tangents_attr = _create_attr(
        driven_key_prim, "weightedTangents", Sdf.ValueTypeNames.Bool
    )
    weighted_tangents_attr.default = driven_key.weighted_tangents.get()

    return driven_key_prim


def get_scene_driven_keys_data() -> Sdf.Layer:
    layer = Sdf.Layer.CreateAnonymous()
    for driven_key in get_scene_driven_keys():
        driven_key_node = AnimCurveNode.from_existing(driven_key)
        prim = define_prim_from_driven_key(layer, f"/{driven_key}", driven_key_node)
    return layer


def merge_driven_key_from_prim(prim: Usd.Prim) -> AnimCurveNode:
    driven_key_name = prim.GetName()

    output_attr = prim.GetAttribute("output")
    output_paths: list[str] = output_attr.Get()

    if cmds.objExists(driven_key_name):
        if not cmds.objectType(driven_key_name, isAType="animCurve"):
            raise RuntimeError(
                f"A node called {driven_key_name} already exists in the scene and isn't an animCurve node."
            )
        driven_key = AnimCurveNode.from_existing(driven_key_name)
    else:
        existing_output = next((path for path in output_paths if cmds.objExists(path)), None)
        if existing_output:
            ouput_plug = get_plug(existing_output)
            curve_fn: MFnAnimCurve = MFnAnimCurve()
            curve_type = curve_fn.unitlessAnimCurveTypeForPlug(ouput_plug)
            curve: MObject = curve_fn.create(curve_type)
            curve_node: MFnDependencyNode = MFnDependencyNode(curve)
            name: str = curve_node.setName(driven_key_name)
            driven_key = AnimCurveNode.from_existing(name)
        else:
            driven_key = AnimCurveUU.create(driven_key_name)

    input_attr = prim.GetAttribute("input")
    input_value: str = input_attr.Get()
    current_input = driven_key.input.get_input()
    current_input = str(current_input) if current_input else ""
    if current_input != input_value:
        try:
            driven_key.input.connect_from(input_value)
        except Exception:
            log.warning(
                f"Couldn't connect {driven_key}'s input to the attribute specified in it's data: {input_value}"
            )

    existing_outputs = {str(output) for output in driven_key.output.get_outputs()}
    to_connect = set(output_paths) - existing_outputs
    to_disconnect = existing_outputs - set(output_paths)
    for output_path in to_connect:
        try:
            driven_key.output.connect_to(output_path)
        except Exception:
            log.warning(
                f"Couldn't connect {driven_key}'s output to the attribute specified in it's data: {output_path}"
            )
    for output_path in to_disconnect:
        driven_key.output.disconnect_from(output_path)

    driven_key_mfn: MFnAnimCurve = MFnAnimCurve(get_depend_node(driven_key))

    key_time_attr = prim.GetAttribute("keyTime")
    key_time_value: list[float] = key_time_attr.Get()

    key_value_attr = prim.GetAttribute("keyValue")
    key_value_value: list[float] = key_value_attr.Get()

    keys = list(zip(key_time_value, key_value_value, strict=True))

    for index in reversed(range(driven_key_mfn.numKeys)):
        driven_key_mfn.remove(index)

    weighted_tangents_attr = prim.GetAttribute("weightedTangents")
    weighted_tangents_value: bool = weighted_tangents_attr.Get()
    driven_key_mfn.setIsWeighted(weighted_tangents_value)

    for driver_value, value in keys:
        driven_key_mfn.addKey(driver_value, value)

    for index in range(driven_key_mfn.numKeys):
        driven_key_mfn.setTangentsLocked(index, False)
        driven_key_mfn.setWeightsLocked(index, False)

    key_tan_in_attr = prim.GetAttribute("keyTanIn")
    key_tan_in_value: Vt.Vec2dArray = key_tan_in_attr.Get()
    for index, key_tan_in in enumerate(key_tan_in_value):
        driven_key_mfn.setTangent(index, key_tan_in[0], key_tan_in[1], True)

    key_tan_in_type_attr = prim.GetAttribute("keyTanInType")
    key_tan_in_type_value: Vt.TokenArray = key_tan_in_type_attr.Get()

    for index, key_tan_in_type in enumerate(key_tan_in_type_value):
        key_tan_in_type_enum = KeyframeTangentType.from_usd_token(key_tan_in_type)
        driven_key_mfn.setInTangentType(index, key_tan_in_type_enum.ktype)

    key_tan_out_attr = prim.GetAttribute("keyTanOut")
    key_tan_out_value: Vt.Vec2dArray = key_tan_out_attr.Get()
    for index, key_tan_out in enumerate(key_tan_out_value):
        driven_key_mfn.setTangent(index, key_tan_out[0], key_tan_out[1], False)

    key_tan_out_type_attr = prim.GetAttribute("keyTanOutType")
    key_tan_out_type_value: Vt.TokenArray = key_tan_out_type_attr.Get()
    for index, key_tan_out_type in enumerate(key_tan_out_type_value):
        key_tan_out_type_enum = KeyframeTangentType.from_usd_token(key_tan_out_type)
        driven_key_mfn.setOutTangentType(index, key_tan_out_type_enum.ktype)

    key_tan_locked_attr = prim.GetAttribute("keyTanLocked")
    key_tan_locked_value: Vt.BoolArray = key_tan_locked_attr.Get()
    for index, key_tan_locked in enumerate(key_tan_locked_value):
        driven_key_mfn.setTangentsLocked(index, key_tan_locked)

    key_weight_locked_attr = prim.GetAttribute("keyWeightLocked")
    key_weight_locked_value: Vt.BoolArray = key_weight_locked_attr.Get()
    for index, key_weight_locked in enumerate(key_weight_locked_value):
        driven_key_mfn.setWeightsLocked(index, key_weight_locked)

    pre_inifinity_attr = prim.GetAttribute("preInfinity")
    pre_inifinity_value: str = pre_inifinity_attr.Get()
    pre_inifinity_enum = KeyframeInfinityType.from_usd_token(pre_inifinity_value)
    driven_key_mfn.setPreInfinityType(pre_inifinity_enum.ktype)

    post_inifinity_attr = prim.GetAttribute("postInfinity")
    post_inifinity_value: str = post_inifinity_attr.Get()
    post_inifinity_enum = KeyframeInfinityType.from_usd_token(post_inifinity_value)
    driven_key_mfn.setPostInfinityType(post_inifinity_enum.ktype)

    return driven_key


def apply_scene_driven_keys_data(stage: Usd.Stage) -> list[AnimCurveNode]:
    applied = []
    for prim in stage.Traverse():
        if (prim.GetTypeName() == "AnimCurve") or prim.IsA("AnimCurve"):
            driven_key = merge_driven_key_from_prim(prim)
            applied.append(driven_key)
    return applied
