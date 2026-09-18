import random

from yrig.color.convert import lch_to_lab, oklab_to_linear_srgb


def random_color_fixed_lightness_chroma(
    lightness: float = 0.65,
    chroma: float = 0.15,
) -> tuple[float, float, float]:
    """
    Generate a random color with fixed lightness and chroma using OKLCH.

    Args:
        lightness: OKLCH lightness in the ``[0.0, 1.0]`` range.
        chroma: OKLCH chroma.

    Returns:
        Linear sRGB color tuple (RGB).
    """
    color = (lightness, chroma, random.uniform(0.0, 360.0))
    return oklab_to_linear_srgb(lch_to_lab(color))
