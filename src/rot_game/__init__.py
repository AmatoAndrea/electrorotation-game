"""
Core package for the new simulator implementation.

This module will grow to include configuration models, asset management,
trajectory planning, and rendering logic. For now it only exposes the
configuration loader while Stage 1 is in progress.
"""

from .config import SimulationConfig, load_simulation_config
from .assets import (
    AssetManager,
    AssetLoadingError,
    BackgroundProvider,
    CellTemplate,
)
from .core.schedule import FrameSchedule, build_frame_schedule
from .core.trajectory import build_per_frame_positions
from .core.renderer import Renderer, GroundTruth, CellGroundTruth
from .core.scaling import radius_um_to_pixels
from .settings import asset_root, output_root, resolve_asset_path, reset_settings_cache
from .exporters import (
    DEFAULT_VIDEO_CODEC,
    determine_scenario_name,
    prepare_output_directory,
    export_video,
    export_ground_truth_csv,
    export_ground_truth_json,
    export_png_frames,
    export_simulation_outputs,
)

__all__ = [
    "SimulationConfig",
    "load_simulation_config",
    "AssetManager",
    "AssetLoadingError",
    "BackgroundProvider",
    "CellTemplate",
    "FrameSchedule",
    "build_frame_schedule",
    "build_per_frame_positions",
    "Renderer",
    "GroundTruth",
    "CellGroundTruth",
    "radius_um_to_pixels",
    "asset_root",
    "output_root",
    "resolve_asset_path",
    "reset_settings_cache",
    "DEFAULT_VIDEO_CODEC",
    "determine_scenario_name",
    "prepare_output_directory",
    "export_video",
    "export_ground_truth_csv",
    "export_ground_truth_json",
    "export_png_frames",
    "export_simulation_outputs",
]
