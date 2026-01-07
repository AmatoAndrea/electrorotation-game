"""Geometry and scaling helpers for the simulator."""

from __future__ import annotations


def radius_um_to_pixels(radius_um: float, magnification: float, pixel_size_um: float) -> float:
    """
    Convert a physical cell radius in micrometers to pixels.

    Parameters
    ----------
    radius_um:
        Physical cell radius in micrometers.
    magnification:
        Microscope magnification factor.
    pixel_size_um:
        Camera pixel size in micrometers.
    """
    if magnification <= 0 or pixel_size_um <= 0:
        raise ValueError("Magnification and pixel_size_um must be positive")
    return (magnification * radius_um) / pixel_size_um
