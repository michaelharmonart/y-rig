def remap(
    input: float,
    input_range: tuple[float, float],
    output_range: tuple[float, float],
) -> float:
    """Linearly remap a value from one range to another.

    Performs a linear interpolation that maps *input* from *input_range* onto
    the corresponding position within *output_range*.  Values outside the
    input range are extrapolated (not clamped).

    Args:
        input: The value to remap.
        input_range: A ``(min, max)`` tuple defining the source range.
        output_range: A ``(min, max)`` tuple defining the destination range.

    Returns:
        The remapped value in *output_range*.

    Raises:
        ZeroDivisionError: If the input range has zero length
            (i.e. ``input_range[0] == input_range[1]``).
    """
    input_range_size = input_range[1] - input_range[0]
    output_range_size = output_range[1] - output_range[0]
    output_value = (
        ((input - input_range[0]) * output_range_size) / input_range_size
    ) + output_range[0]

    return output_value
