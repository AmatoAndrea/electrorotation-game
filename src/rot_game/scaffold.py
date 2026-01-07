"""Scenario scaffolding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .catalog import (
    find_mask_for_template,
    list_backgrounds,
    list_cell_lines,
    list_cell_templates,
)
from .settings import asset_root


DEFAULT_FREQUENCIES = [30, 40, 50]
DEFAULT_DURATIONS = [4.0, 4.0, 4.0]


@dataclass
class ScaffoldChoices:
    background: Path
    cell_template: Path
    mask: Path
    cell_line: Optional[str] = None


def choose_assets(
    background_path: Optional[Path] = None,
    cell_template_path: Optional[Path] = None,
    mask_path: Optional[Path] = None,
    cell_line: Optional[str] = None,
) -> ScaffoldChoices:
    backgrounds = list_backgrounds()
    if not backgrounds:
        raise ValueError("No backgrounds found in asset root")

    available_lines = list_cell_lines()
    if available_lines:
        names = [p.name for p in available_lines]
        if cell_line:
            if cell_line not in names:
                raise ValueError(f"Cell line '{cell_line}' not found in assets")
            cells = list_cell_templates(cell_line)
        else:
            cell_line = names[0]
            cells = list_cell_templates(cell_line)
    else:
        cells = list_cell_templates()
    if not cells:
        raise ValueError("No cell templates found in asset root")

    background = _resolve_preference(background_path, backgrounds)
    cell = _resolve_preference(cell_template_path, cells)

    if mask_path is not None:
        mask = mask_path
    else:
        mask = find_mask_for_template(cell)
        if mask is None:
            raise ValueError(f"No mask found for template {cell.name}")

    return ScaffoldChoices(background=background, cell_template=cell, mask=mask, cell_line=cell_line)


def _resolve_preference(preferred: Optional[Path], options: list[Path]) -> Path:
    if preferred is None:
        return options[0]
    preferred = preferred.resolve()
    if preferred not in options:
        raise ValueError(f"Specified asset not found: {preferred}")
    return preferred


def align_to_frames(value_s: float, fps: float) -> float:
    frames = round(value_s * fps)
    return frames / fps if frames > 0 else 0.0


def write_stub(
    target_path: Path,
    scenario_name: str,
    choices: ScaffoldChoices,
    fps: float = 30.0,
    duration_s: float = 12.0,
    magnification: float = 20.0,
    pixel_size_um: float = 7.4,
) -> None:
    target_path = target_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    background_rel = choices.background.relative_to(asset_root()).as_posix()
    template_rel = choices.cell_template.relative_to(asset_root()).as_posix()
    mask_rel = choices.mask.relative_to(asset_root()).as_posix()

    durations = DEFAULT_DURATIONS.copy()
    total_active = sum(durations)
    if total_active > duration_s:
        durations[-1] = max(duration_s - sum(durations[:-1]), 1.0)
    elif total_active < duration_s:
        durations[-1] = duration_s - sum(durations[:-1])

    durations = [align_to_frames(d, fps) or (1.0 / fps) for d in durations]

    schedule = []
    for freq, dur in zip(DEFAULT_FREQUENCIES, durations):
        schedule.append(
            {
                "frequency_khz": freq,
                "duration_s": float(dur),
                "angular_velocity_rad_s": float(freq / 10.0),
            }
        )

    stub = {
        "video": {
            "resolution": [640, 480],
            "fps": float(fps),
            "duration_s": float(sum(durations)),
            "magnification": float(magnification),
            "pixel_size_um": float(pixel_size_um),
            "background": background_rel,
        },
        "cells": [
            {
                "id": "cell_001",
                "radius_um": 6.0,
                "template": template_rel,
                "mask": {"path": mask_rel},
                "trajectory": {
                    "type": "linear",
                    "start": [100.0, 300.0],
                    "end": [500.0, 200.0],
                },
            }
        ],
        "schedule": {
            "default_delay_s": 0.0,
            "intervals": schedule,
        },
        "metadata": {
            "scenario": scenario_name,
        },
    }

    if choices.cell_line:
        stub["metadata"]["cell_line"] = choices.cell_line

    target_path.write_text(yaml.safe_dump(stub, sort_keys=False), encoding="utf-8")
