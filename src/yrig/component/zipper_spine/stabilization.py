import math
from dataclasses import dataclass

from maya import cmds

from yrig.control.core import Control
from yrig.maya_api.node import (
    DecomposeMatrixNode,
    MultiplyDivideNode,
    MultMatrixNode,
    RemapValueNode,
    SumNode,
)
from yrig.transform.utils import create_transform


@dataclass
class LimitedControl:
    raw_driver: str
    follow: str
    correction: str


def get_relative_matrix(
    driver: str,
    reference: str,
    name: str,
) -> str:

    mult = MultMatrixNode(name=f"{name}_relative_MM")

    mult.matrix_in[0].connect_from(f"{driver}.worldMatrix[0]")

    mult.matrix_in[1].connect_from(f"{reference}.worldInverseMatrix[0]")

    return f"{mult.name}.matrixSum"


def get_relative_position(
    driver: str,
    reference: str,
    name: str,
) -> str:

    matrix = get_relative_matrix(
        driver=driver,
        reference=reference,
        name=name,
    )

    dec = DecomposeMatrixNode(name=f"{name}_relative_DCM")

    dec.input_matrix.connect_from(matrix)

    return f"{dec.name}.outputTranslate"


def get_distance(
    position_a: str,
    position_b: str,
    name: str,
) -> str:

    distance = cmds.createNode(
        "distanceBetween",
        name=f"{name}_DIST",
    )

    cmds.connectAttr(
        position_a,
        f"{distance}.point1",
    )

    cmds.connectAttr(
        position_b,
        f"{distance}.point2",
    )

    return f"{distance}.distance"


def get_distance_ratio(
    position_a: str,
    position_b: str,
    name: str,
) -> tuple[str, float]:

    distance = get_distance(
        position_a=position_a,
        position_b=position_b,
        name=name,
    )

    rest_distance = max(
        cmds.getAttr(distance),
        0.000001,
    )

    ratio = MultiplyDivideNode(name=f"{name}_ratio_MD")

    cmds.setAttr(
        f"{ratio.name}.operation",
        2,  # type:ignore
    )

    ratio.input1.x.connect_from(distance)
    ratio.input2.x.set(rest_distance)

    return f"{ratio.name}.outputX", rest_distance


def get_position_delta(
    position: str,
    name: str,
) -> str:

    rest = cmds.getAttr(position)[0]

    delta = cmds.createNode(
        "plusMinusAverage",
        name=f"{name}_delta_PMA",
    )

    cmds.setAttr(
        f"{delta}.operation",
        2,  # type:ignore
    )

    cmds.connectAttr(
        position,
        f"{delta}.input3D[0]",
    )

    cmds.setAttr(
        f"{delta}.input3D[1]",
        *rest,
        type="float3",
    )

    return f"{delta}.output3D"


def subtract_vectors(
    vector_a: str,
    vector_b: str,
    name: str,
) -> str:

    node = cmds.createNode(
        "plusMinusAverage",
        name=f"{name}_subtract_PMA",
    )

    cmds.setAttr(f"{node}.operation", 2)  # type:ignore

    cmds.connectAttr(
        vector_a,
        f"{node}.input3D[0]",
    )

    cmds.connectAttr(
        vector_b,
        f"{node}.input3D[1]",
    )

    return f"{node}.output3D"


def create_dynamic_soft_weight(
    name: str,
    distance_ratio: str,
    weight: tuple[float, float, float],
    compression_min: float = 0.25,
    compression_max: float = 0.75,
) -> str:

    remap = RemapValueNode(name=f"{name}_compression_RMV")

    remap.input_min.set(compression_min)
    remap.input_max.set(compression_max)

    remap.output_min.set(0.0)
    remap.output_max.set(1.0)

    remap.input_value.connect_from(distance_ratio)

    # Smooth interpolation.
    cmds.setAttr(
        f"{remap.name}.value[0].value_Interp",
        2,  # type:ignore
    )

    mult = MultiplyDivideNode(name=f"{name}_weight_MD")

    mult.input1.set(weight)

    for axis in (
        mult.input2.x,
        mult.input2.y,
        mult.input2.z,
    ):
        axis.connect_from(remap.output)

    return f"{mult.name}.output"


def limit_soft_offset(
    offset: str,
    name: str,
    max_distance: float,
) -> str:

    distance = cmds.createNode(
        "distanceBetween",
        name=f"{name}_offsetLength_DIST",
    )

    cmds.connectAttr(
        offset,
        f"{distance}.point2",
    )

    # Avoid division by zero.
    safe_distance = cmds.createNode(
        "condition",
        name=f"{name}_safeDistance_COND",
    )

    cmds.setAttr(f"{safe_distance}.operation", 2)  # type:ignore

    cmds.connectAttr(
        f"{distance}.distance",
        f"{safe_distance}.firstTerm",
    )

    cmds.setAttr(
        f"{safe_distance}.secondTerm",
        0.000001,  # type:ignore
    )

    cmds.connectAttr(
        f"{distance}.distance",
        f"{safe_distance}.colorIfTrueR",
    )

    cmds.setAttr(
        f"{safe_distance}.colorIfFalseR",
        0.000001,  # type:ignore
    )

    ratio = MultiplyDivideNode(name=f"{name}_limitRatio_MD")

    cmds.setAttr(f"{ratio.name}.operation", 2)  # type:ignore

    ratio.input1.x.set(max_distance)

    ratio.input2.x.connect_from(f"{safe_distance}.outColorR")

    clamp = cmds.createNode(
        "clamp",
        name=f"{name}_limit_CLAMP",
    )

    cmds.setAttr(f"{clamp}.minR", 0)  # type:ignore
    cmds.setAttr(f"{clamp}.maxR", 1)  # type:ignore

    ratio.output.x.connect_to(f"{clamp}.inputR")

    mult = MultiplyDivideNode(name=f"{name}_limitedOffset_MD")

    mult.input1.connect_from(offset)

    for axis in (
        mult.input2.x,
        mult.input2.y,
        mult.input2.z,
    ):
        axis.connect_from(f"{clamp}.outputR")

    return f"{mult.name}.output"


def build_stabilized_drivers(
    name: str,
    drivers: list[str],
    cog: str,
    soft_influence: float = 0.2,
    longitudinal_influence: float = 0.05,
    compression_min: float = 0.25,
    compression_max: float = 0.75,
    max_offset_ratio: float = 0.3,
) -> list[str]:

    reference_grp = cmds.createNode(
        "transform",
        name=f"{name}_reference_grp",
        parent=cog,
    )

    cmds.delete(cmds.orientConstraint(drivers[0], reference_grp, maintainOffset=False))  # type:ignore

    # -----------------------------------
    # Raw COG-relative positions
    # -----------------------------------

    positions = [
        get_relative_position(
            driver=driver,
            reference=reference_grp,
            name=f"{name}_{i:02d}",
        )
        for i, driver in enumerate(drivers)
    ]

    deltas = [
        get_position_delta(
            position=position,
            name=f"{name}_{i:02d}",
        )
        for i, position in enumerate(positions)
    ]

    # -----------------------------------
    # Raw segment measurements
    # -----------------------------------

    segment_ratios = []
    rest_distances = []

    for i in range(len(drivers) - 1):
        ratio, rest = get_distance_ratio(
            position_a=positions[i],
            position_b=positions[i + 1],
            name=f"{name}_segment_{i:02d}",
        )

        segment_ratios.append(ratio)
        rest_distances.append(rest)

    # The reference group is aligned to the rest root's X axis,
    # independently of the COG control's own axes.
    weight = (
        longitudinal_influence,
        soft_influence,
        soft_influence,
    )

    stabilized = []

    # -----------------------------------
    # Stabilized driver layer
    # -----------------------------------

    for i, driver in enumerate(drivers):
        driver_name = f"{name}_{i:02d}_stable"

        # Preserve the original endpoints.
        if i == 0 or i == len(drivers) - 1:
            stabilized.append(driver)
            continue

        stable = cmds.createNode(
            "transform",
            name=driver_name,
            parent=reference_grp,
        )

        # Preserve the original driver matrix
        # in our shared reference space.

        matrix = get_relative_matrix(
            driver=driver,
            reference=reference_grp,
            name=driver_name,
        )

        dec = DecomposeMatrixNode(name=f"{driver_name}_DCM")

        dec.input_matrix.connect_from(matrix)

        for channel in (
            "Rotate",
            "Scale",
            "Shear",
        ):
            cmds.connectAttr(
                f"{dec.name}.output{channel}",
                f"{stable}.{channel.lower()}",
            )

        # -----------------------------------
        # Neighbor influences
        # -----------------------------------

        soft_sum = cmds.createNode(
            "plusMinusAverage",
            name=f"{driver_name}_softSum_PMA",
        )

        neighbors = [
            (i - 1, i - 1),
            (i + 1, i),
        ]

        for index, (
            neighbor,
            segment,
        ) in enumerate(neighbors):
            dynamic_weight = create_dynamic_soft_weight(
                name=f"{driver_name}_{index:02d}",
                distance_ratio=segment_ratios[segment],
                weight=weight,
                compression_min=compression_min,
                compression_max=compression_max,
            )

            relative_delta = subtract_vectors(
                vector_a=deltas[neighbor],
                vector_b=deltas[i],
                name=f"{driver_name}_{index:02d}",
            )

            mult = MultiplyDivideNode(name=f"{driver_name}_{index:02d}_soft_MD")

            mult.input1.connect_from(relative_delta)

            mult.input2.connect_from(dynamic_weight)

            mult.output.connect_to(f"{soft_sum}.input3D[{index}]")

        # -----------------------------------
        # Limit automatic movement
        # -----------------------------------

        max_distance = (
            min(
                rest_distances[i - 1],
                rest_distances[i],
            )
            * max_offset_ratio
        )

        limited_offset = limit_soft_offset(
            offset=f"{soft_sum}.output3D",
            name=driver_name,
            max_distance=max_distance,
        )

        # -----------------------------------
        # Original position + soft offset
        # -----------------------------------

        final_sum = cmds.createNode(
            "plusMinusAverage",
            name=f"{driver_name}_final_PMA",
        )

        cmds.connectAttr(
            positions[i],
            f"{final_sum}.input3D[0]",
        )

        cmds.connectAttr(
            limited_offset,
            f"{final_sum}.input3D[1]",
        )

        cmds.connectAttr(
            f"{final_sum}.output3D",
            f"{stable}.translate",
        )

        stabilized.append(stable)

    return stabilized


def publish_attribute(
    node: str,
    attribute: str,
    source: str,
) -> str:

    plug = f"{node}.{attribute}"

    if not cmds.objExists(plug):
        cmds.addAttr(
            node,
            longName=attribute,
            attributeType="double",
            defaultValue=1.0,
            keyable=False,
        )

    cmds.setAttr(plug, lock=False)

    cmds.connectAttr(
        source,
        plug,
        force=True,
    )

    cmds.setAttr(
        plug,
        channelBox=True,
        lock=True,
    )

    return plug


def get_chain_length_ratio(
    positions: list[str],
    name: str,
) -> str:

    total = SumNode(name=f"{name}_length_SUM")

    for i in range(len(positions) - 1):
        distance = get_distance(
            position_a=positions[i],
            position_b=positions[i + 1],
            name=f"{name}_{i:02d}",
        )

        total.input[i].connect_from(distance)

    rest_length = max(
        total.output.get(),
        0.000001,
    )

    ratio = MultiplyDivideNode(name=f"{name}_lengthRatio_MD")

    cmds.setAttr(
        f"{ratio.name}.operation",
        2,  # type:ignore
    )

    ratio.input1.x.connect_from(total.output)
    ratio.input2.x.set(rest_length)

    return f"{ratio.name}.outputX"


def publish_spine_measurements(
    name: str,
    hybrid_grp: str,
    drivers: list[str],
    spline_pins: list[str],
    cog: str,
) -> None:

    positions = [
        get_relative_position(
            driver=driver,
            reference=cog,
            name=f"{name}_measure_{i:02d}",
        )
        for i, driver in enumerate(drivers)
    ]

    # -----------------------------------
    # Overall compression
    # -----------------------------------

    compression, _ = get_distance_ratio(
        position_a=positions[0],
        position_b=positions[-1],
        name=f"{name}_compression",
    )

    publish_attribute(
        node=hybrid_grp,
        attribute="compression_ratio",
        source=compression,
    )

    # -----------------------------------
    # Per-segment stretch
    # -----------------------------------

    segment_names = [
        "lower_stretch",
        "lower_mid_stretch",
        "upper_mid_stretch",
        "upper_stretch",
    ]

    for i, segment in enumerate(segment_names):
        ratio, _ = get_distance_ratio(
            position_a=positions[i],
            position_b=positions[i + 1],
            name=f"{name}_{segment}",
        )

        publish_attribute(
            node=hybrid_grp,
            attribute=segment,
            source=ratio,
        )

    # -----------------------------------
    # Overall spline stretch
    # -----------------------------------

    pin_positions = [
        get_relative_position(
            driver=pin,
            reference=cog,
            name=f"{name}_pinMeasure_{i:02d}",
        )
        for i, pin in enumerate(spline_pins)
    ]

    stretch = get_chain_length_ratio(
        positions=pin_positions,
        name=f"{name}_overallStretch",
    )

    publish_attribute(
        node=hybrid_grp,
        attribute="stretch_ratio",
        source=stretch,
    )


def create_soft_limit(
    name: str,
    input_value: str,
    minimum: str,
    maximum: str,
    soft_range: str,
) -> str:

    # -----------------------------------
    # Scalar node utilities
    # -----------------------------------

    def connect_or_set(plug: str, value: str | float) -> None:

        if isinstance(value, str):
            cmds.connectAttr(value, plug)
        else:
            cmds.setAttr(plug, value)  # type:ignore

    def multiply(suffix: str, a: str | float, b: str | float, divide: bool = False) -> str:

        node = MultiplyDivideNode(name=f"{name}_{suffix}_MD")

        if divide:
            cmds.setAttr(f"{node.name}.operation", 2)  # type:ignore

        connect_or_set(f"{node.name}.input1X", a)
        connect_or_set(f"{node.name}.input2X", b)

        return f"{node.name}.outputX"

    def condition(
        suffix: str,
        a: str | float,
        b: str | float,
        true: str | float,
        false: str | float,
        operation: int,
    ) -> str:

        node = cmds.createNode(
            "condition",
            name=f"{name}_{suffix}_COND",
        )

        cmds.setAttr(f"{node}.operation", operation)  # type:ignore

        connect_or_set(f"{node}.firstTerm", a)
        connect_or_set(f"{node}.secondTerm", b)
        connect_or_set(f"{node}.colorIfTrueR", true)
        connect_or_set(f"{node}.colorIfFalseR", false)

        return f"{node}.outColorR"

    # Convert wrapper output to a native plug.
    def sum_plug(suffix: str, *values: str | float) -> str:

        node = cmds.createNode(
            "plusMinusAverage",
            name=f"{name}_{suffix}_PMA",
        )

        for i, value in enumerate(values):
            connect_or_set(f"{node}.input1D[{i}]", value)

        return f"{node}.output1D"

    # -----------------------------------
    # Safe dynamic soft range
    # -----------------------------------

    negative_min = multiply("negativeMin", minimum, -1)

    lower_space = sum_plug("lowerSpace", 1.0, negative_min)

    upper_space = sum_plug("upperSpace", maximum, -1.0)

    available_space = condition(
        "availableSpace",
        lower_space,
        upper_space,
        lower_space,
        upper_space,
        4,  # Less than
    )

    effective_soft = condition(
        "effectiveSoft",
        soft_range,
        available_space,
        soft_range,
        available_space,
        4,
    )

    negative_soft = multiply("negativeSoft", effective_soft, -1)

    soft_squared = multiply("softSquared", effective_soft, effective_soft)

    # -----------------------------------
    # Dynamic thresholds
    # -----------------------------------

    lower_start = sum_plug("lowerStart", minimum, effective_soft)

    upper_start = sum_plug("upperStart", maximum, negative_soft)

    # -----------------------------------
    # Rational soft falloff
    # -----------------------------------

    def create_falloff(suffix: str, threshold: str, direction: float) -> str:

        negative_threshold = multiply(
            f"{suffix}_negativeThreshold",
            threshold,
            -1,
        )

        delta = sum_plug(
            f"{suffix}_delta",
            input_value,
            negative_threshold,
        )

        signed_delta = multiply(
            f"{suffix}_direction",
            delta,
            direction,
        )

        excess = condition(
            f"{suffix}_excess",
            signed_delta,
            0,
            signed_delta,
            0,
            2,  # Greater than
        )

        denominator = sum_plug(
            f"{suffix}_denominator",
            effective_soft,
            excess,
        )

        safe_denominator = condition(
            f"{suffix}_safeDenominator",
            denominator,
            0.000001,
            denominator,
            0.000001,
            2,
        )

        return multiply(
            f"{suffix}_falloff",
            soft_squared,
            safe_denominator,
            divide=True,
        )

    lower_falloff = create_falloff("lower", lower_start, -1)

    upper_falloff = create_falloff("upper", upper_start, 1)

    lower_result = sum_plug(
        "lowerResult",
        minimum,
        lower_falloff,
    )

    negative_upper = multiply("negativeUpper", upper_falloff, -1)

    upper_result = sum_plug(
        "upperResult",
        maximum,
        negative_upper,
    )

    # -----------------------------------
    # Select final output
    # -----------------------------------

    lower_output = condition(
        "lower",
        input_value,
        lower_start,
        lower_result,
        input_value,
        4,
    )

    return condition(
        "upper",
        input_value,
        upper_start,
        upper_result,
        lower_output,
        2,
    )


def limit_spine_length(
    name: str,
    drivers: list[str],
    cog: str,
    hybrid_grp: str,
    min_stretch: str,
    max_stretch: str,
    soft_range: str,
) -> list[str]:

    percentages = (0.0, 0.15, 0.5, 0.85, 1.0)

    if len(drivers) != len(percentages):
        raise ValueError("Expected five spine drivers.")

    reference_grp = cmds.createNode(
        "transform",
        name=f"{name}_lengthLimit_grp",
        parent=cog,
    )

    positions = [
        get_relative_position(
            driver=driver,
            reference=reference_grp,
            name=f"{name}_{i:02d}_length",
        )
        for i, driver in enumerate(drivers)
    ]

    # -----------------------------------
    # Current hip-to-chest ratio
    # -----------------------------------

    raw_ratio, rest_distance = get_distance_ratio(
        position_a=positions[0],
        position_b=positions[-1],
        name=f"{name}_rawStretch",
    )

    limited_ratio = create_soft_limit(
        name=f"{name}_softStretch",
        input_value=raw_ratio,
        minimum=min_stretch,
        maximum=max_stretch,
        soft_range=soft_range,
    )

    publish_attribute(
        node=hybrid_grp,
        attribute="raw_stretch_ratio",
        source=raw_ratio,
    )

    publish_attribute(
        node=hybrid_grp,
        attribute="limited_stretch_ratio",
        source=limited_ratio,
    )

    # -----------------------------------
    # Hip-to-chest vector
    # -----------------------------------

    spine_vector = subtract_vectors(
        vector_a=positions[-1],
        vector_b=positions[0],
        name=f"{name}_spineVector",
    )

    direction = cmds.createNode(
        "vectorProduct",
        name=f"{name}_direction_VP",
    )

    cmds.setAttr(f"{direction}.operation", 0)  # type:ignore
    cmds.setAttr(f"{direction}.normalizeOutput", True)  # type:ignore

    cmds.connectAttr(
        spine_vector,
        f"{direction}.input1",
    )

    # Fallback to the rest direction if the
    # hip and chest occupy almost the same position.
    rest_vector = cmds.getAttr(spine_vector)[0]

    rest_length = math.sqrt(sum(value**2 for value in rest_vector))

    if rest_length < 0.000001:
        raise ValueError("Invalid rest spine length.")

    rest_direction = tuple(value / rest_length for value in rest_vector)

    safe_direction = cmds.createNode(
        "condition",
        name=f"{name}_safeDirection_COND",
    )

    cmds.setAttr(f"{safe_direction}.operation", 2)  # type:ignore
    cmds.setAttr(f"{safe_direction}.secondTerm", 0.000001)  # type:ignore

    cmds.connectAttr(
        raw_ratio,
        f"{safe_direction}.firstTerm",
    )

    cmds.connectAttr(
        f"{direction}.output",
        f"{safe_direction}.colorIfTrue",
    )

    cmds.setAttr(
        f"{safe_direction}.colorIfFalse",
        *rest_direction,
        type="float3",
    )

    # -----------------------------------
    # Limited ratio - raw ratio
    # -----------------------------------

    delta = cmds.createNode(
        "plusMinusAverage",
        name=f"{name}_lengthDelta_PMA",
    )

    cmds.setAttr(f"{delta}.operation", 2)  # type:ignore

    cmds.connectAttr(
        limited_ratio,
        f"{delta}.input1D[0]",
    )

    cmds.connectAttr(
        raw_ratio,
        f"{delta}.input1D[1]",
    )

    # Convert ratio difference to scene units.
    distance_delta = MultiplyDivideNode(name=f"{name}_distanceDelta_MD")

    distance_delta.input1.x.connect_from(f"{delta}.output1D")

    distance_delta.input2.x.set(rest_distance)

    # -----------------------------------
    # Full correction vector
    # -----------------------------------

    correction = MultiplyDivideNode(name=f"{name}_correction_MD")

    correction.input1.connect_from(f"{safe_direction}.outColor")

    for axis in (
        correction.input2.x,
        correction.input2.y,
        correction.input2.z,
    ):
        axis.connect_from(distance_delta.output.x)

    # -----------------------------------
    # Distribute correction across drivers
    # -----------------------------------

    limited_drivers = [drivers[0]]

    for i in range(1, len(drivers)):
        driver = drivers[i]
        driver_name = f"{name}_{i:02d}_limited"

        limited = cmds.createNode(
            "transform",
            name=driver_name,
            parent=reference_grp,
        )

        matrix = get_relative_matrix(
            driver=driver,
            reference=reference_grp,
            name=driver_name,
        )

        dec = DecomposeMatrixNode(name=f"{driver_name}_DCM")

        dec.input_matrix.connect_from(matrix)

        for channel in (
            "Rotate",
            "Scale",
            "Shear",
        ):
            cmds.connectAttr(
                f"{dec.name}.output{channel}",
                f"{limited}.{channel.lower()}",
            )

        # Weighted longitudinal correction.
        mult = MultiplyDivideNode(name=f"{driver_name}_correction_MD")

        mult.input1.connect_from(correction.output)

        mult.input2.set((percentages[i],) * 3)  # type:ignore

        # Original position + correction.
        total = cmds.createNode(
            "plusMinusAverage",
            name=f"{driver_name}_position_PMA",
        )

        cmds.connectAttr(
            positions[i],
            f"{total}.input3D[0]",
        )

        mult.output.connect_to(f"{total}.input3D[1]")

        cmds.connectAttr(
            f"{total}.output3D",
            f"{limited}.translate",
        )

        limited_drivers.append(limited)

    return limited_drivers


def create_stretch_attributes(
    hybrid_grp: str,
    min_stretch: float = 0.5,
    max_stretch: float = 1.5,
    soft_range: float = 0.05,
) -> dict[str, str]:

    settings = {
        "min_stretch": (min_stretch, 0.01, 1.0),
        "max_stretch": (max_stretch, 1.0, 10.0),
        "soft_range": (soft_range, 0.0, 1.0),
    }

    attributes = {}

    for attr, (default, minimum, maximum) in settings.items():
        plug = f"{hybrid_grp}.{attr}"

        if not cmds.objExists(plug):
            cmds.addAttr(
                hybrid_grp,
                longName=attr,
                attributeType="double",
                defaultValue=default,
                minValue=minimum,
                maxValue=maximum,
                keyable=False,
            )

        cmds.setAttr(
            plug,
            channelBox=False,
            keyable=False,
            lock=False,
        )

        attributes[attr] = plug

    return attributes


def create_limited_control(
    control: Control,
    follow_driver: str,
    parent: str,
) -> LimitedControl:

    name = control.transform.rsplit("|", 1)[-1]

    # -----------------------------------
    # Unrestricted follow transform
    # -----------------------------------

    follow = create_transform(
        name=f"{name}_rawFollow",
        parent=parent,
        transform=control.offset,
    )

    cmds.parentConstraint(follow_driver, follow, maintainOffset=True)

    # -----------------------------------
    # Visible control correction layer
    # -----------------------------------

    correction = create_transform(
        name=f"{name}_limitOffset",
        parent=follow,
        transform=follow,
    )

    cmds.parent(control.offset, correction)

    # -----------------------------------
    # Independent raw input hierarchy
    # -----------------------------------

    ctrl_parent = cmds.listRelatives(
        control.transform,
        parent=True,
        fullPath=True,
    )[0]

    raw_parent = create_transform(
        name=f"{name}_rawParent",
        parent=follow,
        transform=ctrl_parent,
    )

    raw_driver = cmds.createNode(
        "transform",
        name=f"{name}_rawDriver",
        parent=raw_parent,
    )

    # Read only the LOCAL control matrix.
    # Do not read its world matrix.

    mult = MultMatrixNode(name=f"{name}_rawInput_MM")

    mult.matrix_in[0].connect_from(f"{control.transform}.matrix")

    mult.matrix_in[1].connect_from(f"{control.transform}.offsetParentMatrix")

    cmds.connectAttr(f"{mult.name}.matrixSum", f"{raw_driver}.offsetParentMatrix")

    return LimitedControl(
        raw_driver=raw_driver,
        follow=follow,
        correction=correction,
    )


def connect_limited_control(
    control: LimitedControl,
    limited_driver: str,
    name: str,
) -> None:

    # -----------------------------------
    # Measure both positions in the
    # unrestricted follow space
    # -----------------------------------

    raw_position = get_relative_position(
        driver=control.raw_driver,
        reference=control.follow,
        name=f"{name}_rawPosition",
    )

    limited_position = get_relative_position(
        driver=limited_driver,
        reference=control.follow,
        name=f"{name}_limitedPosition",
    )

    # -----------------------------------
    # Limited position - raw position
    # -----------------------------------

    correction = subtract_vectors(
        vector_a=limited_position,
        vector_b=raw_position,
        name=f"{name}_limitCorrection",
    )

    cmds.connectAttr(
        correction,
        f"{control.correction}.translate",
        force=True,
    )
