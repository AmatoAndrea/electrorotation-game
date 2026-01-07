"""
Frequency schedule expansion utilities.

Transforms the declarative schedule in the configuration into frame-level
arrays that drive rotation and trajectory timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..config import SimulationConfig


@dataclass(frozen=True)
class FrameSchedule:
    """Frame-level schedule information."""

    angular_velocity_rad_per_frame: np.ndarray
    active_mask: np.ndarray
    interval_index: np.ndarray
    active_frame_counts: List[int]
    delay_frame_counts: List[int]

    @property
    def total_frames(self) -> int:
        return int(self.angular_velocity_rad_per_frame.shape[0])

    @property
    def total_active_frames(self) -> int:
        return int(sum(self.active_frame_counts))


def build_frame_schedule(config: SimulationConfig) -> FrameSchedule:
    """
    Expand the schedule into frame-level arrays.

    Raises
    ------
    ValueError
        If angular velocities are missing, or if the schedule duration does not
        match the video duration, or if rounding the durations to frames leaves
        a mismatch.
    """

    fps = float(config.video.fps)
    video_frames = int(round(config.video.duration_s * fps))

    angular_frames: List[float] = []
    active_mask: List[bool] = []
    interval_index: List[int] = []
    active_counts: List[int] = []
    delay_counts: List[int] = []

    total_active_seconds = 0.0
    total_delay_seconds = 0.0

    for idx, interval in enumerate(config.schedule.intervals):
        if interval.angular_velocity_rad_s is None:
            raise ValueError(f"Frequency interval {idx} is missing angular_velocity_rad_s")

        # Determine if this interval is idle (frequency == 0) or active
        is_idle_interval = (interval.frequency_khz == 0.0)
        
        if is_idle_interval:
            # For idle intervals, the entire duration_s is idle (no motion)
            idle_seconds = float(interval.duration_s)
            delay_seconds = (
                float(interval.delay_after_s)
                if interval.delay_after_s is not None
                else float(config.schedule.default_delay_s)
            )
            
            total_delay_seconds += (idle_seconds + delay_seconds)
            
            idle_frames = int(round(idle_seconds * fps))
            delay_frames = int(round(delay_seconds * fps))
            
            # Both are idle - no rotation, no motion
            angular_frames.extend([0.0] * (idle_frames + delay_frames))
            active_mask.extend([False] * (idle_frames + delay_frames))
            interval_index.extend([idx] * (idle_frames + delay_frames))
            
            # For trajectory generation: no active frames, all delay
            active_counts.append(0)
            delay_counts.append(idle_frames + delay_frames)
        else:
            # For active intervals, duration_s is active time
            active_seconds = float(interval.duration_s)
            delay_seconds = (
                float(interval.delay_after_s)
                if interval.delay_after_s is not None
                else float(config.schedule.default_delay_s)
            )

            total_active_seconds += active_seconds
            total_delay_seconds += delay_seconds

            active_frames = int(round(active_seconds * fps))
            delay_frames = int(round(delay_seconds * fps))

            if active_frames <= 0:
                raise ValueError(f"Frequency interval {idx} produces zero active frames")

            rad_per_frame = float(interval.angular_velocity_rad_s) / fps

            angular_frames.extend([rad_per_frame] * active_frames)
            angular_frames.extend([0.0] * delay_frames)

            active_mask.extend([True] * active_frames)
            active_mask.extend([False] * delay_frames)

            interval_index.extend([idx] * (active_frames + delay_frames))

            active_counts.append(active_frames)
            delay_counts.append(delay_frames)

    schedule_seconds = total_active_seconds + total_delay_seconds
    if not np.isclose(schedule_seconds, config.video.duration_s, atol=1e-9):
        raise ValueError(
            "Total schedule duration (including delays) must equal video duration. "
            f"{schedule_seconds} vs {config.video.duration_s}"
        )

    total_frames = len(angular_frames)
    if total_frames != video_frames:
        raise ValueError(
            "Schedule does not align to whole frames at the configured FPS. "
            f"Schedule frames={total_frames}, video frames={video_frames}"
        )

    return FrameSchedule(
        angular_velocity_rad_per_frame=np.asarray(angular_frames, dtype=np.float64),
        active_mask=np.asarray(active_mask, dtype=bool),
        interval_index=np.asarray(interval_index, dtype=np.int32),
        active_frame_counts=active_counts,
        delay_frame_counts=delay_counts,
    )
