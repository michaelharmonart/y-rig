"""
Color utilities for rigging visualization and weight display.

Provides color space conversions (sRGB, Oklab, OkLCH, Rec.2020),
arithmetic helpers for blending and clamping colors, and a spline-based
gradient sampler for generating smooth color ramps.
"""

from . import convert as convert
from . import core as core
from . import gradient as gradient
