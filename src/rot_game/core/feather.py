"""Signed-distance edge feathering utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class FeatherParameters:
    """Runtime parameters for edge feathering."""

    enabled: bool = False
    inside_width_px: float = 0.0
    outside_width_px: float = 0.0

    def is_active(self) -> bool:
        span = self.inside_width_px + self.outside_width_px
        return self.enabled and span > 0.0


def signed_distance_alpha(mask: np.ndarray, params: FeatherParameters) -> np.ndarray:
    """
    Convert a binary (bool) mask to a feathered alpha matte using signed distances.

    Parameters
    ----------
    mask:
        Boolean mask where ``True`` denotes the cell interior.
    params:
        Feathering parameters specifying inside/outside widths.
    """

    if not params.is_active():
        return mask.astype(np.float32)

    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(mask, dtype=np.float32)

    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    signed = inside - outside  # Positive inside, negative outside

    span = params.inside_width_px + params.outside_width_px
    if span <= 0:
        return mask.astype(np.float32)

    t = (signed + params.outside_width_px) / span
    t_clamped = np.clip(t, 0.0, 1.0).astype(np.float32)
    # Smoothstep easing keeps edges soft but bounded in [0, 1]
    eased = t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)
    return eased.astype(np.float32)
