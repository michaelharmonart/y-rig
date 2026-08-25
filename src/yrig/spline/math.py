from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import TypeVar

import numpy as np

from yrig.structs.transform import Vector3

# CV can be anything: a Vector3, a transform name, etc.
CV = TypeVar("CV")
T = TypeVar("T")


def generate_knots(
    count: int, degree: int = 3, periodic: bool = False, clamped: bool = True
) -> list[float]:
    """Generate a default knot vector for a B-spline with the given CV count and degree.

    Args:
        count: total number of UNIQUE points (periodic duplicate points shouldn't be counted).
        degree: The curve degree.  Defaults to ``3`` (cubic).
        periodic: If ``True`` the returned knot vector will be suitable for
            a periodic (closed) curve.  Defaults to ``False``.
        clamped: If ``True`` the returned knot vector will have clamped endpoints (if not periodic).

    Returns:
        list: A list of knot values. (aka knot vector)
    """

    if periodic:
        knots = [(i - degree) for i in range(count + 2 * degree + 1)]
        return [float(knot) for knot in knots]
    if clamped:
        clamp_start = [0] * degree
        clamp_end = [count - degree] * degree
        knots = clamp_start + [i for i in range(count - degree + 1)] + clamp_end
    else:
        knots = [(i - degree) for i in range(count + degree + 1)]

    return [float(knot) for knot in knots]


def is_periodic_knot_vector(knots: Sequence[float], degree: int = 3) -> bool:
    # Based on this equation k[(degree-1)+i+1] - k[(degree-1)+i] = k[(cv_count-1)+i+1] - k[(cv_count)+i]
    # See https://developer.rhino3d.com/guides/opennurbs/periodic-curves-and-surfaces/
    # Although there is a typo in the above doc, k[(cv_count)+i] should be k[(cv_count - 1)+i]
    # Don't ask how long it took me to find that out
    cv_count = len(knots) - (degree + 1)
    for i in range(-degree + 1, degree):
        if (
            knots[(degree - 1) + i + 1] - knots[(degree - 1) + i]
            != knots[(cv_count - 1) + i + 1] - knots[(cv_count - 1) + i]
        ):
            return False
    return True


def create_periodic_cv_list(cvs: Sequence[T], degree: int) -> list[T]:
    """
    Periodic NURBS curves require duplicated CVs at the beginning and end
    of the list to preserve continuity across the seam. This function adds
    the wrapped CVs needed for a curve of the given degree.

    Args:
        cvs: Sequence of unique CV values.
        degree: Curve degree.

    Returns:
        Expanded CV list with wrapped prefix/suffix CVs suitable for
        periodic curve creation.

    Example:
        >>> create_periodic_cv_list([0, 1, 2, 3, 4], degree=3)
        [4, 0, 1, 2, 3, 4, 0]
    """
    shift = degree // 2
    prefix = cvs[-shift:] if shift > 0 else ()
    return list(chain(prefix, cvs, cvs[: degree - shift]))


def collapse_periodic_cv_list(cvs: Sequence[T], degree: int) -> list[T]:
    """
    Remove wrapped periodic CVs and recover the unique CV list.

    This is the inverse operation of ``create_periodic_cv_list`` and assumes
    the input CV sequence follows the same wrapping convention.

    Args:
        cvs: Expanded periodic CV sequence.
        degree: Curve degree used to generate the periodic list.

    Returns:
        List of unique CV values with duplicated periodic CVs removed.

    Example:
        >>> collapse_periodic_cv_list([4, 0, 1, 2, 3, 4, 0], degree=3)
        [0, 1, 2, 3, 4]
    """
    shift = degree // 2
    start = shift
    end = len(cvs) - (degree - shift)
    return list(cvs[start:end])


def find_span(
    t: float,
    knots: Sequence[float],
    degree: int = 3,
) -> int:
    """Find the knot span index containing t.

    Returns span i such that:
        knots[i] <= t < knots[i + 1]
    """
    domain_start = knots[degree]
    domain_end = knots[-degree - 1]

    first_span = degree
    last_span = len(knots) - degree - 2

    if t <= domain_start:
        return first_span

    if t >= domain_end:
        return last_span

    return bisect_right(knots, t) - 1


def deboor_setup(
    cv_count: int,
    t: float,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    normalize: bool = False,
    wrap_parameter: bool = False,
) -> tuple[list[float], int, float]:
    # Algorithm and code originally from Cole O'Brien. Modified to support periodic splines.
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """Prepare the knot vector, span index, and parameter for De Boor evaluation.

    Validates inputs, optionally normalizes the parameter *t* into the
    knot domain, wraps *t* for periodic curves, and locates the knot span
    that contains *t*.  The returned tuple provides everything that
    `deboor_weights` needs to run.

    Args:
        cvs: A sequence of control vertices (only the length is used).
        t: The parameter value at which to evaluate.
        degree: The curve degree.  Defaults to ``3`` (cubic).
        knots: An explicit knot vector.  When ``None`` a uniform clamped
            vector is generated via `generate_knots`.
        normalize: When ``True`` (the default), *t* is treated as a
            ``0``–``1`` value and is remapped to the knot domain before
            span lookup.

    Returns:
        A 4-tuple ``(knots, span, t, periodic)`` where *knots* is the
        (possibly generated) knot list, *span* is the index of the knot
        interval containing *t*, *t* is the (possibly remapped / wrapped)
        parameter, and *periodic* indicates whether the knot vector is
        periodic.

    Raises:
        ValueError: If the number of CVs is too small for the given degree,
            or the knot vector length does not equal ``len(cvs) + degree + 1``.
    """

    order = degree + 1  # Our functions often use order instead of degree
    if cv_count <= degree:
        raise ValueError(f"Curves of degree {degree} require at least {degree + 1} CVs.")

    knots = knots or generate_knots(cv_count, degree)  # Defaults to even knot distribution
    if len(knots) != cv_count + order:
        raise ValueError(
            f"Not enough knots provided. Curves with {cv_count} cvs must have a knot vector of length {cv_count + order}. "
            f"Received a knot vector of length {len(knots)}: {knots}. "
            "Total knot count must equal len(cvs) + degree + 1."
        )

    # Optional normalization of t
    domain_start = knots[degree]
    domain_end = knots[-degree - 1]
    domain_range = domain_end - domain_start

    if normalize:
        t = (t * domain_range) + domain_start

    if wrap_parameter:
        t = ((t - domain_start) % domain_range) + domain_start  # Wrap t into valid domain

    segment = find_span(t, knots, degree)
    return (list(knots), segment, t)


def deboor_weights(
    cvs: Sequence[CV],
    knots: Sequence[float],
    t: float,
    span: int,
    degree: int = 3,
    cv_weights: dict[CV, float] | None = None,
) -> dict[CV, float]:
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """Evaluate De Boor's algorithm to compute per-CV basis weights at a given parameter.

    Runs De Boor's algorithm on the provided
    CVs for the specified knot span and parameter value.  When *cv_weights*
    are supplied the result is the rational (NURBS) basis; otherwise a
    standard B-spline basis is computed (all CV weights equal to ``1``).

    Args:
        cvs: The sequence of control vertices (or their hashable proxies).
        knots: The full knot vector.
        t: The parameter value at which to evaluate (must lie within the
            span defined by *span*).
        span: The knot span index containing *t*, as returned by
            :func:`deboor_setup`.
        degree: The curve degree.  Defaults to ``3`` (cubic).
        cv_weights: An optional mapping of ``{cv: weight}`` for rational
            NURBS evaluation.  When ``None``, every CV is assigned a
            weight of ``1`` (pure B-spline).

    Returns:
        A dictionary mapping each contributing CV to its normalised basis
        weight.

    Raises:
        ZeroDivisionError: If the total of all basis weights is zero
            (degenerate configuration).
    """
    if cv_weights is None:
        cv_weights = {cv: 1 for cv in cvs}

    # Run a modified version of de Boors algorithm
    cv_bases = [{cv: 1.0} for cv in cvs]  # initialize basis weights with a value of 1 for every cv
    for r in range(1, degree + 1):  # Loop once per degree
        for j in range(degree, r - 1, -1):  # Loop backwards from degree to r
            right = j + 1 + span - r
            left = j + span - degree
            alpha = (t - knots[left]) / (
                knots[right] - knots[left]
            )  # Alpha is how much influence comes from the left vs right cv

            weights = {}
            for cv, weight in cv_bases[j].items():
                weights[cv] = weight * alpha

            for cv, weight in cv_bases[j - 1].items():
                if cv in weights:
                    weights[cv] += weight * (1 - alpha)
                else:
                    weights[cv] = weight * (1 - alpha)

            cv_bases[j] = weights
    final_bases = cv_bases[degree]

    # Multiply each CV's basis function by its weight
    # see: https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline#General_form_of_a_NURBS_curve
    numerator = {i: final_bases[i] * cv_weights[i] for i in final_bases}

    # Sum all of the weights to normalize them such that they all total to 1
    denominator: float = sum(numerator.values())
    if denominator == 0:
        raise ZeroDivisionError("Zero sum of total weight values, unable to normalize.")

    # Actually do the normalization
    rational_weights = {i: numerator[i] / denominator for i in numerator}

    return rational_weights


def point_on_spline_weights(
    cvs: Sequence[CV],
    t: float,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    normalize: bool = True,
    return_zero_weights: bool = False,
) -> list[tuple[CV, float]]:
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Creates a mapping of cvs to curve weight values on a spline curve.
    While all cvs are required, only the cvs with non-zero weights will be returned.
    This function is based on de Boor's algorithm for evaluating splines and has been modified to consolidate weights.
    Args:
        cvs: A list of cvs, these are used for the return value.
        t: A parameter value.
        degree: The curve dimensions.
        knots: A list of knot values.
        weights: A list of CV weight values.
        normalize: When true, the curve is parameter is normalized from 0-1
        return_zero_weights: When True, weights that don't contribute will also be returned.
    Returns:
        list: A list of control point, weight pairs.
    """

    new_knots, segment, mapped_t = deboor_setup(
        cv_count=len(cvs), t=t, degree=degree, knots=knots, normalize=normalize
    )

    # Convert cvs into hash-able indices
    cvs_ids: list[int] = [i for i in range(len(cvs))]
    if weights:
        cv_weights = {cvs_ids[i]: weights[i] for i in range(len(cvs_ids))}
    else:
        cv_weights = None

    # Filter out cvs we won't be using
    filtered_ids = [cvs_ids[j + segment - degree] for j in range(degree + 1)]

    # Run a modified version of de Boors algorithm
    out_weights = deboor_weights(
        cvs=filtered_ids,
        t=mapped_t,
        span=segment,
        degree=degree,
        knots=new_knots,
        cv_weights=cv_weights,
    )

    return [
        (cvs[index], weight)
        for index, weight in reversed(out_weights.items())
        if (weight != 0.0) or return_zero_weights
    ]


def get_weights_along_spline(
    cvs: Sequence[CV],
    parameters: Sequence[float],
    degree: int = 3,
    knots: Sequence[float] | None = None,
    sample_points: int = 128,
) -> list[list[tuple[CV, float]]]:
    """Batch-evaluate B-spline basis weights for many parameter values efficiently.

    When the number of parameters exceeds *sample_points* this function
    builds a lookup table (LUT) of evenly spaced weight samples and
    linearly interpolates between them, which is significantly faster than
    calling `point_on_spline_weights` in a tight loop — especially
    for dense meshes with thousands of vertices.

    When the parameter count is small enough the function falls back to
    exact per-parameter evaluation automatically.

    Args:
        cvs: An ordered sequence of control vertices.  The CV objects are
            returned verbatim in the output tuples.
        parameters: The parameter values at which to evaluate weights.
        degree: Degree of the B-spline.  Defaults to ``3`` (cubic).
        knots: An explicit knot vector.  When ``None`` a uniform clamped
            vector is generated via :func:`generate_knots`.
        sample_points: Number of evenly spaced samples used to build the
            LUT.  Higher values yield more accurate interpolation at the
            cost of a slightly longer setup phase.  Defaults to ``128``.

    Returns:
        A list with one entry per parameter.  Each entry is a list of
        ``(cv, weight)`` tuples (same format as
        :func:`point_on_spline_weights`).
    """
    cv_ids = list(range(len(cvs)))

    if not knots:
        knots = generate_knots(len(cv_ids), degree=degree)

    result: list[list[tuple[CV, float]]] = []
    # If we have less points than samples don't bother using a lookup table
    if len(parameters) <= sample_points:
        for parameter in parameters:
            sample_weights: list[tuple[CV, float]] = point_on_spline_weights(
                cvs=cvs, t=parameter, degree=degree, knots=knots, normalize=False
            )
            result.append(sample_weights)
        return result

    # Precompute lookup table
    parameter_array = np.array(parameters, dtype=float)
    min_t, max_t = min(parameters), max(parameters)
    t_range: float = max_t - min_t
    if t_range == 0:
        # All parameters are the same, just calculate the one weight
        single_weights = point_on_spline_weights(
            cvs=cvs, t=min_t, degree=degree, knots=knots, normalize=False
        )
        return [single_weights for _ in parameters]

    # Get evenly spaced points from the minimum to maximum t value
    sample_params = np.linspace(min_t, max_t, sample_points, dtype=float)
    lut_weights = np.zeros((sample_points, len(cv_ids)), dtype=float)

    for sample_index, sample_parameter in enumerate(sample_params):
        weights: list[tuple[int, float]] = point_on_spline_weights(
            cvs=cv_ids, t=sample_parameter, degree=degree, knots=knots, normalize=False
        )
        weight_dict = {cv_id: w for cv_id, w in weights}
        # Take the weights and put them into the correct row in the array
        lut_weights[sample_index, :] = [weight_dict.get(cv_id, 0.0) for cv_id in cv_ids]

    # Map each parameter to LUT index positions
    normalized_positions = (parameter_array - min_t) / t_range * (sample_points - 1)
    lower_indices = np.floor(normalized_positions).astype(int)
    upper_indices = np.clip(lower_indices + 1, 0, sample_points - 1)
    interpolation_alphas = (normalized_positions - lower_indices)[:, None]

    # Interpolate weights for all parameters in bulk
    interpolated_weight_array = (1 - interpolation_alphas) * lut_weights[
        lower_indices, :
    ] + interpolation_alphas * lut_weights[upper_indices, :]

    # Reattach CV references to each interpolated weight row
    for weight_row in interpolated_weight_array:
        result.append(list(zip(cvs, weight_row.tolist(), strict=True)))
    return result


def tangent_on_spline_weights(
    cvs: Sequence[CV],
    t: float,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    normalize: bool = True,
) -> list[tuple[CV, float]]:
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367

    # This cannot be used for full NURBS, only B-Splines (NURBS where every CV has a weight of 1)
    # as the derivative of a full NURB Spline cannot be expressed as a weighted sum of point positions
    """Compute per-CV weights for the **tangent** (first derivative) of a B-spline at a parameter.

    The tangent of a degree-*d* B-spline can be expressed as a weighted
    sum of CV positions where the weights come from a degree-*(d − 1)*
    basis evaluated on a modified knot vector.  This function returns
    those weights so that the caller can compute the tangent vector as
    ``sum(cv_position * weight for cv_position, weight in result)``.

    .. note::
        This function only supports **B-splines** (all CV weights equal to ``1``).
        It cannot be used for rational NURBS curves because the
        derivative of a rational curve is not a simple weighted sum of CV
        positions.

    Args:
        cvs: An ordered sequence of control vertices.  The CV objects are
            returned in the output tuples.
        t: The parameter value at which to evaluate the tangent.
        degree: The curve degree.  Defaults to ``3`` (cubic).
        knots: An explicit knot vector.  When ``None`` a uniform clamped
            vector is generated via :func:`generate_knots`.
        normalize: When ``True`` (the default), *t* is treated as a
            ``0``–``1`` value and is remapped to the knot domain.

    Returns:
        A list of ``(cv, weight)`` tuples for CVs with non-zero tangent
        contribution at *t*.
    """

    new_knots, segment, t = deboor_setup(
        cv_count=len(cvs), t=t, degree=degree, knots=knots, normalize=normalize
    )

    # Convert cvs into hash-able indices
    cv_ids: list[int] = [i for i in range(len(cvs))]

    # In order to find the tangent we need to find points on a lower degree curve
    lower_degree: int = degree - 1
    weights = deboor_weights(cvs=cv_ids, t=t, span=segment, degree=lower_degree, knots=new_knots)

    # Take the lower order weights and match them to our actual cvs
    remapped_weights: list[tuple[int, float]] = []
    for j in range(lower_degree + 1):
        weight: float = weights[j]
        cv0: int = j + segment - lower_degree
        cv1: int = j + segment - lower_degree - 1
        alpha: float = (
            weight
            * (lower_degree + 1)
            / (new_knots[j + segment + 1] - new_knots[j + segment - lower_degree])
        )
        remapped_weights.append((cv_ids[cv0], alpha))
        remapped_weights.append((cv_ids[cv1], -alpha))

    # Add weights of corresponding CVs and only return those that are > 0
    deduplicated_weights = {i: 0.0 for i in cv_ids}
    for item in remapped_weights:
        deduplicated_weights[item[0]] += item[1]
    deduplicated_weights = {key: value for key, value in deduplicated_weights.items() if value != 0}

    return [(cvs[index], weight) for index, weight in deduplicated_weights.items()]


def get_point_on_spline(
    cv_positions: Sequence[Vector3],
    t: float,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    normalize_parameter: bool = True,
) -> Vector3:
    """Evaluate the position on a B-spline (or NURBS) curve at parameter *t*.

    Computes basis weights via :func:`point_on_spline_weights` and returns
    the weighted sum of CV positions.

    Args:
        cv_positions: Ordered control-vertex positions as
            :class:`~yrig.structs.transform.Vector3` instances.
        t: The parameter value at which to evaluate.
        degree: The curve degree.  Defaults to ``3`` (cubic).
        knots: An explicit knot vector, or ``None`` for auto-generation.
        weights: Per-CV rational weights for NURBS.  ``None`` for a pure
            B-spline.
        normalize_parameter: When ``True``, *t* is in the ``0``–``1``
            range.

    Returns:
        A :class:`~yrig.structs.transform.Vector3` representing the
        world-space point on the curve.
    """
    position: Vector3 = Vector3()
    for control_point, weight in point_on_spline_weights(
        cvs=cv_positions,
        t=t,
        degree=degree,
        knots=knots,
        weights=weights,
        normalize=normalize_parameter,
    ):
        position += control_point * weight
    return position


def get_tangent_on_spline(
    cv_positions: Sequence[Vector3], t: float, degree: int = 3, knots: Sequence[float] | None = None
) -> Vector3:
    """Evaluate the tangent vector of a B-spline curve at parameter *t*.

    Computes tangent basis weights via :func:`tangent_on_spline_weights`
    and returns the weighted sum of CV positions, yielding the first
    derivative of the curve.

    Args:
        cv_positions: Ordered control-vertex positions as
            :class:`~yrig.structs.transform.Vector3` instances.
        t: The parameter value at which to evaluate.
        degree: The curve degree.  Defaults to ``3`` (cubic).
        knots: An explicit knot vector, or ``None`` for auto-generation.

    Returns:
        A :class:`~yrig.structs.transform.Vector3` representing the
        (unnormalised) tangent direction at *t*.
    """
    tangent: Vector3 = Vector3()
    for control_point, weight in tangent_on_spline_weights(
        cvs=cv_positions, t=t, degree=degree, knots=knots
    ):
        tangent += control_point * weight
    return tangent


def _get_arc_length_table(samples: Iterable[Vector3]) -> list[float]:
    arc_lengths: list[float] = [0.0]
    samples = iter(samples)
    prev = next(samples)
    for sample in samples:
        arc_lengths.append(arc_lengths[-1] + (sample - prev).length())
        prev = sample
    return arc_lengths


def _invert_arc_length(
    arc_lengths: Sequence[float],
    sample_params: Sequence[float],
    target_length: float,
) -> float:
    """
    Map an arc-length value back to a curve parameter using a sampled lookup
    table and linear interpolation.
    """
    index = bisect_left(arc_lengths, target_length)

    if index <= 0:
        return sample_params[0]
    if index >= len(arc_lengths):
        return sample_params[-1]

    prev_index = index - 1
    prev_length, length = arc_lengths[prev_index], arc_lengths[index]

    sample_distance = length - prev_length
    alpha = (target_length - prev_length) / sample_distance if sample_distance else 0.0
    interpolated = sample_params[prev_index] + alpha * (
        sample_params[index] - sample_params[prev_index]
    )
    return interpolated


def resample(
    cv_positions: Sequence[Vector3],
    number_of_points: int,
    degree: int = 3,
    knots: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    periodic: bool = False,
    padded: bool = True,
    arc_length: bool = True,
    sample_points: int = 256,
    u_start: float | None = None,
    u_end: float | None = None,
    normalize_parameter: bool = True,
) -> list[float]:
    """Compute evenly spaced parameter values along a B-spline curve.

    Given a set of CV positions this function returns *number_of_points*
    parameter values that are either uniformly distributed in parameter
    space or uniformly distributed by **arc length** (the default).

    Arc-length resampling works by densely sampling the curve, building a
    cumulative-distance lookup table, and binary-searching for the
    parameter that corresponds to each desired fractional distance.

    Args:
        cv_positions: Ordered CV positions as Vector3 instances.
        number_of_points: How many evenly spaced sample parameters to produce.
        degree: The curve degree.  Defaults to 3 (cubic).
        knots: An explicit knot vector.  When None a uniform clamped vector is generated.
        weights: Per-CV rational weights for NURBS. None for a pure B-spline.
        periodic: When True, samples are distributed for a periodic (closed) curve.
        padded: When True (and periodic is False), the first and
            last samples are inset by half a segment width from the curve
            endpoints, which avoids placing joints exactly at the tips.
            Defaults to True.
        arc_length: When True, the returned parameters
            are evenly spaced by arc length rather than by raw parameter value.
        sample_points: The number of dense samples used internally to
            approximate arc length.  Higher values yield more accurate
            spacing.  Defaults to 256 which is plenty.
        u_start: Optional start of the resampling range (in the same
            coordinate system as *normalize_parameter* implies).  Defaults
            to the domain start.
        u_end: Optional end of the resampling range.  Defaults to the
            domain end.
        normalize_parameter: When True (the default), the returned
            parameters and *u_min* / *u_max* are in the ``0``–``1`` range.
            When ``False``, raw knot-domain values are used.

    Returns:
        A list of float parameter values, ordered from the start to the end of the curve.
    """

    if not knots:
        new_knots = generate_knots(count=len(cv_positions), degree=degree, periodic=periodic)
    else:
        new_knots = knots

    domain_start, domain_end = (
        (0, 1) if normalize_parameter else (new_knots[degree], new_knots[-degree - 1])
    )

    calculated_u_start = u_start if u_start is not None else domain_start
    calculated_u_max = u_end if u_end is not None else domain_end
    if not calculated_u_start < calculated_u_max:
        raise ValueError(
            f"The start U value ({calculated_u_start}) must be less than the max U value ({calculated_u_max})"
        )

    def get_normalized_u(index: int) -> float:
        if periodic:
            if padded:
                base_u = (index + 0.5) / (number_of_points)
            else:
                base_u = index / (number_of_points)
        else:
            if padded:
                base_u = (index + 0.5) / number_of_points
            else:
                base_u = index / (number_of_points - 1)
        return base_u

    def get_target_u(index: int) -> float:
        return calculated_u_start + (calculated_u_max - calculated_u_start) * get_normalized_u(
            index
        )

    if not arc_length:
        return [get_target_u(i) for i in range(number_of_points)]

    # Arc length based resampling
    if sample_points < 2:
        raise ValueError("sample_points must be >= 2")

    sample_params: list[float] = [
        calculated_u_start + (calculated_u_max - calculated_u_start) * (i / (sample_points - 1))
        for i in range(sample_points)
    ]
    samples: list[Vector3] = [
        get_point_on_spline(
            cv_positions=cv_positions,
            t=param,
            degree=degree,
            knots=new_knots,
            weights=weights,
            normalize_parameter=normalize_parameter,
        )
        for param in sample_params
    ]

    arc_lengths: list[float] = _get_arc_length_table(samples)
    total_length: float = arc_lengths[-1]
    point_parameters: list[float] = []

    for i in range(number_of_points):
        normalized_u = get_normalized_u(i)
        target_length: float = normalized_u * total_length
        mapped_t: float = _invert_arc_length(arc_lengths, sample_params, target_length)
        point_parameters.append(mapped_t)

    return point_parameters
