"""Edge feathering configuration shared between GUI and rendering layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...core.feather import FeatherParameters


@dataclass
class EdgeFeatherConfig:
    """Session-scoped configuration for mask edge feathering."""

    enabled: bool = False
    inside_pixels: float = 3.0
    outside_pixels: float = 3.0
    min_pixels: float = 1.0
    max_pixels: float = 8.0

    def _clamp_pixels(self, value: float) -> float:
        return max(self.min_pixels, min(self.max_pixels, value))

    def set_inside_pixels(self, value: float) -> None:
        self.inside_pixels = self._clamp_pixels(value)

    def set_outside_pixels(self, value: float) -> None:
        self.outside_pixels = self._clamp_pixels(value)

    def inside_microns(self, effective_um_per_px: Optional[float]) -> Optional[float]:
        if not effective_um_per_px or effective_um_per_px <= 0:
            return None
        return self.inside_pixels * effective_um_per_px

    def outside_microns(self, effective_um_per_px: Optional[float]) -> Optional[float]:
        if not effective_um_per_px or effective_um_per_px <= 0:
            return None
        return self.outside_pixels * effective_um_per_px

    def to_runtime_params(self, _: float | None = None) -> FeatherParameters:
        return FeatherParameters(
            enabled=self.enabled,
            inside_width_px=self.inside_pixels,
            outside_width_px=self.outside_pixels,
        )
