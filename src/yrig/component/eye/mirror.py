from maya.api.OpenMaya import MMatrix


def mirror_matrix(
    matrix: MMatrix,
    axis: str = "x",
    behavior: bool = True,
) -> MMatrix:
    """Mirror a world matrix across a plane through the origin.

    Args:
        matrix: Source world-space matrix.
        axis: Axis perpendicular to the mirror plane.
        behavior: If True, mirror rotation behavior while
            preserving positive handedness.

    Returns:
        Mirrored world-space matrix.
    """

    axis = axis.lower()

    if axis not in ("x", "y", "z"):
        raise ValueError(f"Invalid mirror axis: {axis}")

    axis_index = "xyz".index(axis)

    reflection = [1.0, 1.0, 1.0]
    reflection[axis_index] = -1.0

    reflection_matrix = MMatrix(
        [
            reflection[0],
            0.0,
            0.0,
            0.0,
            0.0,
            reflection[1],
            0.0,
            0.0,
            0.0,
            0.0,
            reflection[2],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )

    if behavior:
        # Reflect local and world axes.
        # The two reflections preserve handedness.
        return reflection_matrix * matrix * reflection_matrix

    # Reflect world position only, preserving rotation.
    values = list(matrix)  # type:ignore
    values[12 + axis_index] *= -1.0

    return MMatrix(values)


def mirror_eye_matrix(matrix: MMatrix) -> MMatrix:
    """Mirror a right-side eye matrix's orientation.

    Assumes the matrix is already positioned on the right side
    of the character (negative world X).

    Uses X-axis reflection on both sides of the rotation
    matrix to preserve right-handedness.

    The original world-space translation is retained.
    """

    reflection = MMatrix(
        (
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )

    # Mirror rotation behavior.
    mirrored = reflection * matrix * reflection

    # Preserve the existing right-side position.
    values = list(mirrored)
    values[12] = matrix[12]
    values[13] = matrix[13]
    values[14] = matrix[14]

    return MMatrix(values)
