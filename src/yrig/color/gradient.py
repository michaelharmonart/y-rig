from dataclasses import dataclass

from yrig.color.core import blend_colors_by_weight
from yrig.spline.math import point_on_spline_weights


@dataclass
class GradientStop:
    """A single color stop within a :class:`Gradient`.

    Attributes:
        position: The normalized position of this stop along the gradient,
            typically in the range ``[0.0, 1.0]``.
        color: The color value at this stop, represented as a tuple of three
            floats (e.g. OkLCH, Oklab, or RGB depending on context).
    """

    position: float
    color: tuple[float, float, float]


@dataclass
class Gradient:
    """A spline-based color gradient defined by a sequence of color stops.

    The gradient is evaluated by treating the stop positions as B-spline
    control-vertex parameters and blending the associated colors using
    De Boor's algorithm at the requested degree.

    Attributes:
        stops: An ordered tuple of :class:`GradientStop` instances that
            define the gradient's color ramp.
        degree: The B-spline degree used when interpolating between stops.
    """

    stops: tuple[GradientStop, ...]
    degree: int


OKLCH_HEATMAP_GRADIENT = Gradient(
    stops=(
        GradientStop(0.0, (0, 0.05, -80)),
        GradientStop(0.01, (0.1, 0.15, -60)),
        GradientStop(0.2, (0.7, 0.15, 0)),
        GradientStop(1.0, (1.0, 0.15, 90)),
    ),
    degree=2,
)


def get_gradient_knots(gradient: Gradient):
    """Generate a clamped knot vector for the given gradient.

    Builds a knot vector suitable for B-spline evaluation over the gradient's
    stops.  The vector is clamped at both ends so that the spline passes
    through the first and last stops, with internal knots averaged from
    neighbouring stop positions (Schoenberg–Whitney style).

    Args:
        gradient: The :class:`Gradient` whose stop positions and degree are
            used to construct the knot vector.

    Returns:
        list[float]: A clamped knot vector of length
        ``len(gradient.stops) + gradient.degree + 1``.
    """
    degree = gradient.degree
    stop_positions = [stop.position for stop in gradient.stops]
    clamp_start = [stop_positions[0]] * (degree + 1)
    clamp_end = [stop_positions[-1]] * (degree + 1)

    num_internal = len(gradient.stops) - degree - 1
    internal_knots = list(
        (sum(stop_positions[j : j + degree]) / degree) for j in range(1, num_internal + 1)
    )
    clamped_knots = clamp_start + internal_knots + clamp_end
    return [float(knot) for knot in clamped_knots]


def sample_spline_gradient(gradient: Gradient, position: float) -> tuple[float, float, float]:
    """Sample a color from the gradient at the given position.

    Evaluates the B-spline defined by the gradient's color stops and knot
    vector at *position*, returning a blended color.

    Args:
        gradient: The :class:`Gradient` to sample from.
        position: The position along the gradient at which to evaluate.
            Should lie within the range defined by the first and last stop
            positions (typically ``0.0`` to ``1.0``).

    Returns:
        A three-component color tuple produced by blending the gradient
        stop colors using the computed spline basis weights.
    """
    gradient_stop_colors = [stop.color for stop in gradient.stops]
    gradient_knots = get_gradient_knots(gradient)
    color_weights = point_on_spline_weights(
        cvs=gradient_stop_colors,
        t=position,
        knots=gradient_knots,
        degree=gradient.degree,
        normalize=False,
    )
    return blend_colors_by_weight(
        colors=(color_weight[0] for color_weight in color_weights),
        weights=(color_weight[1] for color_weight in color_weights),
    )
