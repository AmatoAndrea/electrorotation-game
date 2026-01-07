"""Color utility functions."""

from __future__ import annotations

import colorsys
import math
from typing import Tuple

# Frequency range used to map colors (kHz); values outside clamp to the ends.
_MIN_FREQ_KHZ = 10.0
_MAX_FREQ_KHZ = 200_000.0
_MIN_LOG = math.log(_MIN_FREQ_KHZ)
_MAX_LOG = math.log(_MAX_FREQ_KHZ)

# HSV parameters for the warm→cool ramp (red to blue).
_START_HUE = 0.0  # red
_END_HUE = 220.0 / 360.0  # blue
_SATURATION = 0.65
_VALUE = 0.95
_ALPHA = 90


def frequency_brush(freq: float) -> Tuple[int, int, int, int]:
    """Return RGBA tuple representing the frequency on a log-scaled warm→cool ramp."""
    if freq <= 0:
        clamped = _MIN_FREQ_KHZ
    else:
        clamped = min(max(freq, _MIN_FREQ_KHZ), _MAX_FREQ_KHZ)
    # Avoid divide-by-zero if min == max (defensive, though constants differ)
    if _MAX_LOG == _MIN_LOG:
        ratio = 0.0
    else:
        ratio = (math.log(clamped) - _MIN_LOG) / (_MAX_LOG - _MIN_LOG)
    hue = _START_HUE + ratio * (_END_HUE - _START_HUE)
    r, g, b = colorsys.hsv_to_rgb(hue, _SATURATION, _VALUE)
    return int(r * 255), int(g * 255), int(b * 255), _ALPHA
