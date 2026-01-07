"""
Configuration models and loader for the simulator.

The simulator uses YAML scenario files as the single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, conint, confloat, validator, root_validator

from .settings import resolve_asset_path


PositiveFloat = confloat(gt=0)
NonNegativeFloat = confloat(ge=0)


class VideoConfig(BaseModel):
    """Global parameters for video generation."""

    resolution: Tuple[conint(gt=0), conint(gt=0)] = Field(
        default=(640, 480), description="(width, height) in pixels"
    )
    fps: PositiveFloat = Field(..., description="Frames per second of the output video")
    duration_s: PositiveFloat = Field(..., description="Total video duration in seconds")
    magnification: PositiveFloat = Field(..., description="Microscope magnification factor")
    pixel_size_um: PositiveFloat = Field(
        ..., description="Physical size of one camera pixel in micrometers"
    )
    background: Optional[Path] = Field(
        default=None, description="Path to background image (static)"
    )
    background_frames_dir: Optional[Path] = Field(
        default=None, description="Directory containing ordered background frames"
    )
    background_ref_mag: PositiveFloat = Field(
        default=10.93, description="Reference magnification at which the background image was captured"
    )
    noise_enabled: bool = Field(
        default=False, description="Enable quantization noise to simulate camera read noise"
    )
    noise_stddev: PositiveFloat = Field(
        default=5.0, description="Standard deviation of Gaussian noise (typical: 2-10)"
    )

    @validator("background")
    def _validate_background(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        if not value.exists():
            raise ValueError(f"Background file not found: {value}")
        return value

    @validator("background_frames_dir")
    def _validate_background_frames_dir(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return value
        if not value.exists():
            raise ValueError(f"Background frames directory not found: {value}")
        if not value.is_dir():
            raise ValueError(f"Background frames path is not a directory: {value}")
        return value

    @root_validator
    def _require_one_background_source(cls, values: dict) -> dict:
        image = values.get("background")
        frames_dir = values.get("background_frames_dir")
        if (image is None) == (frames_dir is None):
            raise ValueError("Exactly one of background or background_frames_dir must be provided")
        return values


class TrajectoryConfig(BaseModel):
    """
    Defines the motion path for a cell.

    Trajectories are specified in pixel coordinates relative to the video frame.
    
    Built-in trajectory types: "linear", "parabolic", "cubic", "stationary"
    
    Custom trajectory types can be registered via plugins using the
    `rot_game.core.trajectory.register_trajectory_generator` function.
    Custom types may use additional fields via the `params` dictionary.
    """

    type: str = Field(
        default="linear",
        description="Trajectory type (built-in: linear, parabolic, cubic, stationary; or custom plugin types)",
    )
    start: Optional[Tuple[float, float]] = Field(
        default=None, description="Starting (x, y) coordinates in pixels"
    )
    end: Optional[Tuple[float, float]] = Field(
        default=None, description="Ending (x, y) coordinates in pixels"
    )
    position: Optional[Tuple[float, float]] = Field(
        default=None, description="Stationary position coordinates (x, y)"
    )
    control_points: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="Optional control points for curved trajectories (interpreted per type)",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for custom trajectory types (plugin-specific)",
    )

    @root_validator
    def _validate_coordinates(cls, values: dict) -> dict:
        traj_type = values.get("type")
        start = values.get("start")
        end = values.get("end")
        position = values.get("position")
        
        # Built-in types that require specific validation
        builtin_types = {"linear", "parabolic", "cubic", "stationary"}

        if traj_type == "stationary":
            if position is None:
                position = start or end
            if position is None:
                raise ValueError("Stationary trajectory requires 'position'")
            values["position"] = position
            values["start"] = position
            values["end"] = position
            values["control_points"] = None
        elif traj_type in builtin_types:
            # Built-in non-stationary types require start and end
            if start is None or end is None:
                raise ValueError(f"{traj_type} trajectory requires 'start' and 'end'")
            values["position"] = None
        else:
            # Custom trajectory type - allow flexible validation
            # Plugins are responsible for validating their own parameters
            # At minimum, we need a position reference (start, end, or position)
            if start is None and end is None and position is None:
                raise ValueError(
                    f"Custom trajectory type '{traj_type}' requires at least one of: "
                    "'start', 'end', or 'position'"
                )
            # Default start to position if not provided
            if start is None and position is not None:
                values["start"] = position
        return values

    def dict(self, *args, **kwargs) -> dict:  # type: ignore[override]
        data = super().dict(*args, **kwargs)
        if data.get("type") == "stationary":
            data.pop("start", None)
            data.pop("end", None)
        else:
            data.pop("position", None)
        # Remove params if empty
        if data.get("params") is None:
            data.pop("params", None)
        return data


class MaskConfig(BaseModel):
    """Mask resource required to extract the cell foreground from the template."""

    path: Path = Field(..., description="Path to a binary mask image matching the template resolution")

    @validator("path")
    def _validate_mask_path(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Mask file not found: {value}")
        return value


class CellConfig(BaseModel):
    """Configuration for a single simulated cell."""

    id: str = Field(..., description="Unique cell identifier in the scenario")
    radius_um: PositiveFloat = Field(..., description="Physical radius of the cell in micrometers")
    template: Path = Field(..., description="Path to the cell template image (grayscale)")
    mask: MaskConfig = Field(..., description="Mask configuration for removing background from the template")
    trajectory: TrajectoryConfig

    @validator("template")
    def _validate_template(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Cell template file not found: {value}")
        return value


class FrequencyInterval(BaseModel):
    """
    Represents an electrophoretic frequency interval.
    """

    frequency_khz: NonNegativeFloat = Field(
        ..., description="Frequency applied during the interval (0 allowed for idle segments)"
    )
    duration_s: PositiveFloat = Field(..., description="Duration of the interval in seconds")
    angular_velocity_rad_s: Optional[float] = Field(
        default=None,
        description="Expected angular velocity in radians per second for the interval",
    )
    delay_after_s: Optional[NonNegativeFloat] = Field(
        default=None,
        description="Optional delay after the interval during which cells stay still. Uses schedule "
        "default when omitted.",
    )


class ScheduleConfig(BaseModel):
    """Complete frequency schedule along with global defaults."""

    default_delay_s: NonNegativeFloat = Field(
        default=0.0, description="Default delay between frequency changes if not overridden"
    )
    intervals: List[FrequencyInterval]

    @validator("intervals")
    def _validate_intervals(cls, value: Iterable[FrequencyInterval]) -> List[FrequencyInterval]:
        intervals = list(value)
        if not intervals:
            raise ValueError("At least one frequency interval must be defined")
        return intervals


class SimulationConfig(BaseModel):
    """Top-level configuration object for a simulation scenario."""

    video: VideoConfig
    cells: List[CellConfig]
    schedule: ScheduleConfig
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for bookkeeping"
    )

    @validator("cells")
    def _validate_cells(cls, value: Iterable[CellConfig]) -> List[CellConfig]:
        cells = list(value)
        if not cells:
            raise ValueError("The scenario must include at least one cell")
        return cells


def _resolve_relative_paths(data: dict, base_path: Path) -> dict:
    """
    Replace relative paths in the raw configuration dictionary with absolute paths.

    The function mutates the dictionary in-place for convenience.
    """

    def _maybe_resolve(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
        if path_value is None:
            return None
        path_obj = Path(path_value)
        candidates: List[Path] = []

        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.append((base_path / path_obj).resolve())
            candidates.append(resolve_asset_path(path_obj))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    video = data.get("video")
    if isinstance(video, dict):
        if "background" in video:
            video["background"] = _maybe_resolve(video["background"])
        if "background_frames_dir" in video:
            video["background_frames_dir"] = _maybe_resolve(video["background_frames_dir"])

    for cell in data.get("cells", []):
        if "template" in cell:
            cell["template"] = _maybe_resolve(cell["template"])

        mask = cell.get("mask")
        if isinstance(mask, dict) and "path" in mask:
            mask["path"] = _maybe_resolve(mask["path"])

    return data


def load_simulation_config(path: Union[str, Path]) -> SimulationConfig:
    """
    Load and validate a simulation scenario from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    SimulationConfig
        Parsed and validated configuration object.
    """

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    processed = _resolve_relative_paths(raw_data, config_path.parent)
    return SimulationConfig.parse_obj(processed)
