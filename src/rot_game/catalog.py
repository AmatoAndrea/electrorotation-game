"""Asset catalog utilities for the simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from .settings import asset_root


IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def _collect(paths: Iterable[Path]) -> List[Path]:
    return sorted({path.resolve() for path in paths if path.exists()})


def list_backgrounds() -> List[Path]:
    root = asset_root() / "backgrounds"
    if not root.exists():
        raise FileNotFoundError(f"Background directory not found: {root}")
    candidates: List[Path] = []
    for pattern in IMAGE_GLOBS:
        candidates.extend(root.glob(pattern))
    paths = _collect(candidates)
    if not paths:
        raise FileNotFoundError(f"No background images found in {root}")
    return paths

def list_cell_lines() -> List[Path]:
    root = asset_root() / "cell_lines"
    if not root.exists():
        raise FileNotFoundError(f"Cell lines directory not found: {root}")
    lines = sorted([path for path in root.iterdir() if path.is_dir()])
    if not lines:
        raise FileNotFoundError(f"No cell lines found in {root}")
    return lines


def list_cell_templates(cell_line: Optional[str] = None) -> List[Path]:
    root_lines = asset_root() / "cell_lines"
    if not root_lines.exists():
        raise FileNotFoundError(f"Cell lines directory not found: {root_lines}")

    candidates: List[Path] = []

    if cell_line:
        base = root_lines / cell_line / "cells"
        if not base.exists():
            raise FileNotFoundError(f"Cells directory not found for line '{cell_line}': {base}")
        for pattern in IMAGE_GLOBS:
            candidates.extend(base.glob(pattern))
    else:
        for line_dir in list_cell_lines():
            base = line_dir / "cells"
            for pattern in IMAGE_GLOBS:
                candidates.extend(base.glob(pattern))
    paths = _collect(candidates)
    if not paths:
        raise FileNotFoundError("No cell templates found; ensure each cell line has a 'cells/' directory with images")
    return paths


def list_masks(cell_line: Optional[str] = None) -> List[Path]:
    root_lines = asset_root() / "cell_lines"
    if not root_lines.exists():
        raise FileNotFoundError(f"Cell lines directory not found: {root_lines}")

    candidates: List[Path] = []
    if cell_line:
        base = root_lines / cell_line / "masks"
        if not base.exists():
            raise FileNotFoundError(f"Masks directory not found for line '{cell_line}': {base}")
        for pattern in IMAGE_GLOBS:
            candidates.extend(base.glob(pattern))
    else:
        for line_dir in list_cell_lines():
            base = line_dir / "masks"
            for pattern in IMAGE_GLOBS:
                candidates.extend(base.glob(pattern))
    return _collect(candidates)

def find_mask_for_template(template_path: Path) -> Optional[Path]:
    template_name = template_path.stem
    line_name = None
    parts = template_path.parts
    if "cell_lines" in parts:
        idx = parts.index("cell_lines")
        if idx + 1 < len(parts):
            line_name = parts[idx + 1]
    possible_masks = list_masks(line_name)
    for mask_path in possible_masks:
        if mask_path.stem.startswith(template_name):
            return mask_path
    return None
