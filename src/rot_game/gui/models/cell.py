"""Cell scenario data model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class ScenarioCell:
    """Represents a cell in the simulation scenario.
    
    Attributes:
        id: Unique identifier for the cell
        template: Path to the cell template image
        mask: Path to the cell mask image
        trajectory_type: Type of trajectory (linear, parabolic, cubic, stationary)
        start: Starting position (x, y) in pixels
        end: Ending position (x, y) in pixels
        control_points: List of control points for Bézier curves
        params: Additional trajectory-specific parameters
    
    Note:
        Cell radius is calculated dynamically from the mask, not stored here.
    """
    id: str
    template: Path
    mask: Path
    trajectory_type: str = "stationary"
    start: Tuple[float, float] = (320.0, 240.0)
    end: Tuple[float, float] = (320.0, 240.0)
    control_points: List[Tuple[float, float]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def ensure_controls(
        self,
        resolution: Tuple[int, int] | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Ensure control_points list has the correct length for the trajectory type.
        
        - linear/stationary: no control points
        - parabolic: 1 control point
        - cubic: 2 control points
        """
        if self.trajectory_type == "parabolic":
            if force or not self.control_points:
                control = self._default_parabolic_control(resolution)
                self.control_points = [control]
            elif len(self.control_points) > 1:
                self.control_points = [self.control_points[0]] if not force else [
                    self._default_parabolic_control(resolution)
                ]
        elif self.trajectory_type == "cubic":
            if force or not self.control_points:
                self.control_points = list(self._default_cubic_controls(resolution))
            elif len(self.control_points) == 1:
                if force:
                    self.control_points = list(self._default_cubic_controls(resolution))
                else:
                    # Preserve existing first point, add a symmetric partner
                    _, second = self._default_cubic_controls(resolution)
                    self.control_points.append(second)
            elif len(self.control_points) > 2:
                self.control_points = self.control_points[:2]
        else:
            # linear or stationary: no control points
            self.control_points = []

    def default_end_for_resolution(self, resolution: Tuple[int, int]) -> Tuple[float, float]:
        """Compute the default end point given the current start and resolution.
        
        The end point is placed along the vector from start toward the image center,
        with magnitude equal to one quarter of the image width. If start is at the
        center, fall back to a rightward vector of the same magnitude.
        """
        start_x, start_y = self.start
        width, height = resolution
        center = (float(width) / 2.0, float(height) / 2.0)
        target_vec = (center[0] - start_x, center[1] - start_y)
        length = math.hypot(target_vec[0], target_vec[1])
        desired = float(width) / 4.0

        if length == 0:
            # Start is at center: push to the right
            return (start_x + desired, start_y)

        scale = desired / length
        end_x = start_x + target_vec[0] * scale
        end_y = start_y + target_vec[1] * scale
        return (end_x, end_y)

    def _default_parabolic_control(self, resolution: Tuple[int, int] | None) -> Tuple[float, float]:
        """Place a single control point perpendicular to the segment midpoint."""
        midpoint = self._midpoint()
        offset = self._perpendicular_offset(
            self._segment_length(),
            prefer_center=self._center_from_resolution(resolution),
        )
        return (midpoint[0] + offset[0], midpoint[1] + offset[1])

    def _default_cubic_controls(
        self,
        resolution: Tuple[int, int] | None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Place two control points equally spaced and offset perpendicular to the path."""
        dx, dy = self._segment_vector()
        step = (dx / 3.0, dy / 3.0)
        offset = self._perpendicular_offset_vector(self._segment_length())

        c1_base = (self.start[0] + step[0], self.start[1] + step[1])
        c2_base = (self.start[0] + 2 * step[0], self.start[1] + 2 * step[1])

        control1 = (c1_base[0] + offset[0], c1_base[1] + offset[1])
        control2 = (c2_base[0] - offset[0], c2_base[1] - offset[1])
        return control1, control2

    def _segment_vector(self) -> Tuple[float, float]:
        return (self.end[0] - self.start[0], self.end[1] - self.start[1])

    def _segment_length(self) -> float:
        dx, dy = self._segment_vector()
        return math.hypot(dx, dy)

    def _midpoint(self) -> Tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0)

    def _perpendicular_offset_vector(self, magnitude: float) -> Tuple[float, float]:
        dx, dy = self._segment_vector()
        length = math.hypot(dx, dy)
        if length == 0 or magnitude == 0:
            return (0.0, 0.0)
        scale = magnitude / length
        return (-dy * scale, dx * scale)

    def _perpendicular_offset(
        self,
        magnitude: float,
        *,
        prefer_center: Tuple[float, float] | None = None,
    ) -> Tuple[float, float]:
        """Return a perpendicular offset with optional center-aware sign selection."""
        base = self._perpendicular_offset_vector(magnitude)
        if prefer_center is None:
            return base
        midpoint = self._midpoint()
        candidate1 = (midpoint[0] + base[0], midpoint[1] + base[1])
        candidate2 = (midpoint[0] - base[0], midpoint[1] - base[1])
        dist1 = math.hypot(candidate1[0] - prefer_center[0], candidate1[1] - prefer_center[1])
        dist2 = math.hypot(candidate2[0] - prefer_center[0], candidate2[1] - prefer_center[1])
        return base if dist1 <= dist2 else (-base[0], -base[1])

    @staticmethod
    def _center_from_resolution(resolution: Tuple[int, int] | None) -> Tuple[float, float] | None:
        if resolution is None:
            return None
        width, height = resolution
        return (float(width) / 2.0, float(height) / 2.0)
