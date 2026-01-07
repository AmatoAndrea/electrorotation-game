"""Application state data model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ...config import (
    CellConfig,
    FrequencyInterval,
    MaskConfig,
    ScheduleConfig,
    SimulationConfig,
    TrajectoryConfig,
    VideoConfig,
)
from ..utils import calculate_radius_from_mask
from .cell import ScenarioCell
from .edge_feather import EdgeFeatherConfig
from .schedule import ScheduleInterval


@dataclass
class ScenarioState:
    """Represents the complete state of a simulation scenario.
    
    Attributes:
        background: Path to background image
        cells: List of cells in the scenario
        schedule: List of frequency schedule intervals
        resolution: Video resolution (width, height) in pixels
        fps: Frames per second
        magnification: Microscope magnification
        pixel_size_um: Physical size of one pixel in micrometers
        video_duration_s: Total video duration in seconds
        background_ref_mag: Reference magnification for the background image (default: 10.93)
    """
    background: Optional[Path] = None
    background_frames_dir: Optional[Path] = None
    cells: List[ScenarioCell] = field(default_factory=list)
    schedule: List[ScheduleInterval] = field(default_factory=list)  # Start with empty schedule
    resolution: Tuple[int, int] = (640, 480)
    fps: float = 30.0
    magnification: float = 10.93
    pixel_size_um: float = 7.4
    video_duration_s: float = 12.0  # Not used for automatic padding anymore
    background_ref_mag: float = 10.93
    noise_enabled: bool = False
    noise_stddev: float = 5.0
    edge_feather: EdgeFeatherConfig = field(default_factory=EdgeFeatherConfig)

    def has_background_source(self) -> bool:
        if self.background and self.background.exists():
            return True
        if self.background_frames_dir and self.background_frames_dir.exists():
            return True
        return False

    def to_config(self) -> Tuple[SimulationConfig, Optional[float]]:
        """Convert the GUI state to a simulation configuration.
        
        Returns:
            Tuple of (SimulationConfig, None) - second value kept for API compatibility
            but idle intervals are no longer automatically added.
            
        Raises:
            ValueError: If background or cells are not set
        """
        if (self.background is None) and (self.background_frames_dir is None):
            raise ValueError("Background not set")
        if self.background is not None and self.background_frames_dir is not None:
            raise ValueError("Specify either a background image or a background frame folder, not both")
        if not self.cells:
            raise ValueError("No cells in scenario")

        intervals: List[ScheduleInterval] = []
        total_frames = 0
        for interval in self.schedule:
            active_frames = math.ceil(interval.duration_s * self.fps)
            delay_frames = math.ceil(interval.delay_after_s * self.fps)
            total_frames += active_frames + delay_frames
            quantized_duration = active_frames / self.fps
            quantized_delay = delay_frames / self.fps
            intervals.append(
                ScheduleInterval(
                    frequency_khz=interval.frequency_khz,
                    duration_s=quantized_duration,
                    angular_velocity_rad_s=interval.angular_velocity_rad_s,
                    delay_after_s=quantized_delay,
                )
            )

        # Video duration exactly matches schedule duration (no automatic padding)
        actual_duration_s = total_frames / self.fps if total_frames > 0 else 0.0

        video_cfg = VideoConfig(
            resolution=list(self.resolution),
            fps=self.fps,
            duration_s=actual_duration_s,
            magnification=self.magnification,
            pixel_size_um=self.pixel_size_um,
            background=self.background,
            background_frames_dir=self.background_frames_dir,
            background_ref_mag=self.background_ref_mag,
            noise_enabled=self.noise_enabled,
            noise_stddev=self.noise_stddev,
        )

        cell_configs: List[CellConfig] = []
        for cell in self.cells:
            # Calculate radius from mask
            radius_um = calculate_radius_from_mask(cell.mask)
            
            # Get params if available
            cell_params = getattr(cell, 'params', None) or None
            
            if cell.trajectory_type == "stationary":
                trajectory_cfg = TrajectoryConfig(
                    type="stationary",
                    position=list(cell.start),
                    params=cell_params,
                )
            else:
                trajectory_cfg = TrajectoryConfig(
                    type=cell.trajectory_type,
                    start=list(cell.start),
                    end=list(cell.end),
                    control_points=[list(cp) for cp in cell.control_points] if cell.control_points else None,
                    params=cell_params,
                )
            mask_cfg = MaskConfig(path=cell.mask)
            cell_cfg = CellConfig(
                id=cell.id,
                radius_um=radius_um,
                template=cell.template,
                mask=mask_cfg,
                trajectory=trajectory_cfg,
            )
            cell_configs.append(cell_cfg)

        freq_intervals = [
            FrequencyInterval(
                frequency_khz=interval.frequency_khz,
                duration_s=interval.duration_s,
                angular_velocity_rad_s=interval.angular_velocity_rad_s,
                delay_after_s=interval.delay_after_s,
            )
            for interval in intervals
        ]
        schedule_cfg = ScheduleConfig(default_delay_s=0.0, intervals=freq_intervals)

        metadata = {"scenario": self.cells[0].id if self.cells else "scenario"}
        metadata["edge_feather"] = {
            "enabled": self.edge_feather.enabled,
            "inside_pixels": self.edge_feather.inside_pixels,
            "outside_pixels": self.edge_feather.outside_pixels,
        }

        config = SimulationConfig(
            video=video_cfg,
            cells=cell_configs,
            schedule=schedule_cfg,
            metadata=metadata,
        )
        return config, None  # Return None for idle_duration (no longer used)
