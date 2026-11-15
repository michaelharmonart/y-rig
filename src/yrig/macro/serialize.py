from maya import cmds
from pxr import Ar, Sdf, Usd

from yrig.maya_api.node.anim_curve import AnimCurveNode


def _create_attr(
    prim: Usd.Prim, name: Ar.ResolvedPath | str, type_name: Sdf.ValueTypeName
) -> Usd.Attribute:
    return prim.CreateAttribute(name, type_name, custom=False)


def get_scene_driven_keys() -> list[str]:
    return cmds.ls(type=["animCurveUA", "animCurveUL", "animCurveUT", "animCurveUU"]) or []


def define_prim_from_driven_key(
    stage: Usd.Stage, path: Sdf.Path | str, driven_key: AnimCurveNode
) -> Usd.Prim:
    driven_key_prim = stage.DefinePrim(path, typeName="AnimCurve")

    input_attr = _create_attr(driven_key_prim, "input", Sdf.ValueTypeNames.String)
    input_attr.Set(str(driven_key.output.get_input() or ""))

    output_attr = _create_attr(driven_key_prim, "output", Sdf.ValueTypeNames.StringArray)
    output_attr.Set([str(out) for out in driven_key.output.get_outputs()])

    key_time_attr = _create_attr(driven_key_prim, "keyTime", Sdf.ValueTypeNames.DoubleArray)
    key_time_attr.Set([attr.key_time.get() for attr in driven_key.key_time_value])

    key_value_attr = _create_attr(driven_key_prim, "keyValue", Sdf.ValueTypeNames.DoubleArray)
    key_value_attr.Set([attr.key_time.get() for attr in driven_key.key_time_value])

    key_tan_in_attr = _create_attr(driven_key_prim, "keyTanIn", Sdf.ValueTypeNames.Double2Array)
    key_tan_in_attr.Set(
        [
            (x.get(), y.get())
            for x, y in zip(driven_key.key_tan_in_x, driven_key.key_tan_in_y, strict=True)
        ]
    )

    key_tan_in_type_attr = _create_attr(
        driven_key_prim, "keyTanInType", Sdf.ValueTypeNames.TokenArray
    )
    key_tan_in_type_attr.Set(token.get().usd_token for token in driven_key.key_tan_in_type)

    key_tan_out_attr = _create_attr(driven_key_prim, "keyTanOut", Sdf.ValueTypeNames.Double2Array)
    key_tan_out_attr.Set(
        [
            (x.get(), y.get())
            for x, y in zip(driven_key.key_tan_in_x, driven_key.key_tan_in_y, strict=True)
        ]
    )

    key_tan_out_type_attr = _create_attr(
        driven_key_prim, "keyTanOutType", Sdf.ValueTypeNames.TokenArray
    )
    key_tan_out_type_attr.Set(token.get().usd_token for token in driven_key.key_tan_out_type)

    key_tan_locked_attr = _create_attr(
        driven_key_prim, "keyTanLocked", Sdf.ValueTypeNames.BoolArray
    )
    key_tan_locked_attr.Set([attr.get() for attr in driven_key.key_tan_locked])

    key_weight_locked_attr = _create_attr(
        driven_key_prim, "keyWeightLocked", Sdf.ValueTypeNames.BoolArray
    )
    key_weight_locked_attr.Set([attr.get() for attr in driven_key.key_weight_locked])

    pre_inifinity_attr = _create_attr(driven_key_prim, "preInfinity", Sdf.ValueTypeNames.Token)
    pre_inifinity_attr.Set(driven_key.pre_infinity.get().usd_token)

    post_inifinity_attr = _create_attr(driven_key_prim, "postInfinity", Sdf.ValueTypeNames.Token)
    post_inifinity_attr.Set(driven_key.post_infinity.get().usd_token)

    tangent_type_attr = _create_attr(driven_key_prim, "tangentType", Sdf.ValueTypeNames.Token)
    tangent_type_attr.Set(driven_key.tangent_type.get().usd_token)

    weighted_tangents_attr = _create_attr(
        driven_key_prim, "weightedTangents", Sdf.ValueTypeNames.Bool
    )
    weighted_tangents_attr.Set(driven_key.weighted_tangents.get())

    rotation_interpolation_attr = _create_attr(
        driven_key_prim, "rotationInterpolation", Sdf.ValueTypeNames.Token
    )
    rotation_interpolation_attr.Set(driven_key.rotation_interpolation.get().usd_token)

    return driven_key_prim


def get_scene_driven_keys_data() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    for driven_key in get_scene_driven_keys():
        driven_key_node = AnimCurveNode.from_existing(driven_key)
        prim = define_prim_from_driven_key(stage, f"/{driven_key}", driven_key_node)
    return stage
