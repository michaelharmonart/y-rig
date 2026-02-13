from typing import Iterable


def add_colors(
    color1: tuple[float, float, float], color2: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Add two colors component-wise.

    Args:
        color1: First color as an ``(R, G, B)`` tuple (or any three-component
            color space such as Oklab).
        color2: Second color in the same space as *color1*.

    Returns:
        A new tuple with each channel summed: ``(c1 + c2)`` per component.
    """
    return (color1[0] + color2[0], color1[1] + color2[1], color1[2] + color2[2])


def scale_color(color: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    """Scale every component of a color by a scalar value.

    Args:
        color: The color to scale as a three-component tuple.
        scalar: The multiplier applied to each component.

    Returns:
        A new tuple with each channel multiplied by *scalar*.
    """
    return (color[0] * scalar, color[1] * scalar, color[2] * scalar)


def clamp_value(value: float) -> float:
    """Clamp a single floating-point value to the ``[0.0, 1.0]`` range.

    Args:
        value: The value to clamp.

    Returns:
        *value* clamped so that ``0.0 <= result <= 1.0``.
    """
    return max(0.0, min(1.0, value))


def clamp_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Clamps each component of a color to the range [0.0, 1.0].

    Args:
        color: A tuple of three floats (e.g., RGB or any color space).
    Returns:
        A tuple with each component clamped to the [0.0, 1.0] range.
    """
    return (clamp_value(color[0]), clamp_value(color[1]), clamp_value(color[2]))


def blend_colors_by_weight(colors: Iterable[tuple[float, float, float]], weights: Iterable[float]):
    """Compute a weighted sum of colors.

    Each color is scaled by its corresponding weight and the results are
    accumulated into a single blended color.  When the weights sum to ``1.0``
    the result is a proper weighted average (e.g. for visualising skinning
    influence).

    Args:
        colors: An iterable of three-component color tuples, all in the same
            color space.
        weights: An iterable of scalar weights, paired element-wise with
            *colors*.

    Returns:
        A three-component tuple representing the blended color.
    """
    final_color = (0.0, 0.0, 0.0)
    for color, weight in zip(colors, weights):
        final_color = add_colors(scale_color(color, weight), final_color)
    return final_color
