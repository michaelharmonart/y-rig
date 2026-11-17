from maya import cmds
from pxr import Sdf

from yrig.maya_api.node.anim_curve import AnimCurveNode


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
    key_value_attr.default = [attr.key_time.get() for attr in driven_key.key_time_value]

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

    tangent_type_attr = _create_attr(driven_key_prim, "tangentType", Sdf.ValueTypeNames.Token)
    tangent_type_attr.default = driven_key.tangent_type.get().usd_token

    weighted_tangents_attr = _create_attr(
        driven_key_prim, "weightedTangents", Sdf.ValueTypeNames.Bool
    )
    weighted_tangents_attr.default = driven_key.weighted_tangents.get()

    rotation_interpolation_attr = _create_attr(
        driven_key_prim, "rotationInterpolation", Sdf.ValueTypeNames.Token
    )
    rotation_interpolation_attr.default = driven_key.rotation_interpolation.get().usd_token

    return driven_key_prim


def get_scene_driven_keys_data() -> Sdf.Layer:
    layer = Sdf.Layer.CreateAnonymous()
    for driven_key in get_scene_driven_keys():
        driven_key_node = AnimCurveNode.from_existing(driven_key)
        prim = define_prim_from_driven_key(layer, f"/{driven_key}", driven_key_node)
    return layer
