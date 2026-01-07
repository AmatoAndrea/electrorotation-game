"""Output exporters for simulator artefacts.

Provides helpers for writing rendered frames to video files, ground truth
tables, JSON manifests, and optional PNG frame stacks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .config import SimulationConfig
from .settings import output_root as default_output_root
from .core.renderer import GroundTruth

DEFAULT_VIDEO_CODEC = "XVID"


def determine_scenario_name(config: SimulationConfig, fallback: str = "scenario") -> str:
    """
    Determine a filesystem-friendly scenario name.

    Preference order:
    1. `config.metadata["scenario"]`
    2. `config.metadata["name"]`
    3. Provided fallback string
    """
    for key in ("scenario", "name"):
        value = config.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_name(value)
    return _sanitize_name(fallback)


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip().lower())
    return safe or "scenario"


def prepare_output_directory(output_root: Path, scenario_name: str, timestamp: Optional[str] = None) -> Path:
    """
    Create the directory where all artefacts for a simulation run will be stored.
    """
    if timestamp is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        ts = timestamp

    output_dir = output_root / "simulated_video" / ts
    counter = 1
    while output_dir.exists():
        output_dir = output_root / "simulated_video" / f"{ts}_{counter}"
        counter += 1

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def export_video(frames: np.ndarray, fps: float, output_path: Path, codec: str = DEFAULT_VIDEO_CODEC) -> None:
    """
    Write the rendered frames to an `.avi` file.
    """
    if frames.ndim != 3:
        raise ValueError("Expected frames array with shape (frames, height, width)")

    total_frames, height, width = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)
    if not writer.isOpened():
        raise IOError(f"Failed to open video writer at {output_path}")

    for frame in frames:
        writer.write(frame)
    writer.release()


def export_ground_truth_csv(ground_truth: GroundTruth, output_path: Path) -> None:
    """
    Write ground-truth data to a CSV file with one row per cell per frame.
    """
    fieldnames = ["frame", "cell_id", "x_px", "y_px", "omega_rad_s", "angle_rad", "area_px"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_idx in range(ground_truth.cells[0].positions.shape[0]):
            for cell in ground_truth.cells:
                writer.writerow(
                    {
                        "frame": frame_idx,
                        "cell_id": cell.cell_id,
                        "x_px": float(cell.positions[frame_idx, 0]),
                        "y_px": float(cell.positions[frame_idx, 1]),
                        "omega_rad_s": float(cell.angular_velocity_rad_s[frame_idx]),
                        "angle_rad": float(cell.cumulative_angle_rad[frame_idx]),
                        "area_px": int(cell.area_px[frame_idx]),
                    }
                )


def export_ground_truth_json(
    ground_truth: GroundTruth,
    output_path: Path,
    config: SimulationConfig,
    feather: Optional[dict] = None,
    feather_pixels: int = 0,
) -> None:
    """
    Write a JSON file containing simulation metadata (no per-frame ground truth).
    """
    video_block = {
        "resolution": list(config.video.resolution),
        "fps": ground_truth.fps,
        "duration_s": config.video.duration_s,
        "magnification": config.video.magnification,
        "pixel_size_um": config.video.pixel_size_um,
        "background": str(config.video.background) if config.video.background is not None else None,
        "background_frames_dir": str(config.video.background_frames_dir) if config.video.background_frames_dir is not None else None,
        "noise_enabled": bool(config.video.noise_enabled),
        "noise_stddev": float(config.video.noise_stddev),
    }
    video_block = {k: v for k, v in video_block.items() if v is not None}

    payload = {
        "video": video_block,
        "schedule": {
            "default_delay_s": config.schedule.default_delay_s,
            "intervals": [
                {
                    "frequency_khz": interval.frequency_khz,
                    "duration_s": interval.duration_s,
                    "angular_velocity_rad_s": interval.angular_velocity_rad_s,
                    "delay_after_s": interval.delay_after_s,
                }
                for interval in config.schedule.intervals
            ],
        },
        "feathering": feather or {"feather_pixels": int(feather_pixels)},
        "metadata": config.metadata,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_png_frames(frames: np.ndarray, frames_dir: Path, prefix: str = "frame_", digits: int = 4) -> None:
    """
    Save individual frames as PNG images.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    template = f"{prefix}{{:0{digits}d}}.png"
    for idx, frame in enumerate(frames):
        frame_path = frames_dir / template.format(idx)
        cv2.imwrite(str(frame_path), frame)


def export_simulation_outputs(
    frames: np.ndarray,
    ground_truth: GroundTruth,
    config: SimulationConfig,
    output_root: Optional[Path] = None,
    include_frames: bool = False,
    codec: str = DEFAULT_VIDEO_CODEC,
    feather_params: Optional["FeatherParameters"] = None,
    feather_pixels: int = 0,
    scenario_name: Optional[str] = None,
) -> Path:
    """
    Export all default artefacts for a simulation run.

    Returns the path to the directory containing the artefacts.
    """
    if output_root is None:
        output_root = default_output_root()
    else:
        output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenario_name = scenario_name or determine_scenario_name(config, fallback="scenario")
    output_dir = prepare_output_directory(output_root, scenario_name)

    video_path = output_dir / f"{scenario_name}.avi"
    csv_path = output_dir / f"{scenario_name}_ground_truth.csv"
    json_path = output_dir / "simulation_metadata.json"

    export_video(frames, ground_truth.fps, video_path, codec=codec)
    export_ground_truth_csv(ground_truth, csv_path)
    # Minimal FeatherParameters typing shim to avoid import cycle
    feather_payload = None
    if feather_params is not None:
        feather_payload = {
            "enabled": bool(getattr(feather_params, "enabled", True)),
            "inside_width_px": float(getattr(feather_params, "inside_width_px", 0.0)),
            "outside_width_px": float(getattr(feather_params, "outside_width_px", 0.0)),
            "feather_pixels": int(feather_pixels),
        }
    export_ground_truth_json(ground_truth, json_path, config, feather=feather_payload, feather_pixels=feather_pixels)

    if include_frames:
        frames_dir = output_dir / "frames"
        export_png_frames(frames, frames_dir)

    return output_dir
