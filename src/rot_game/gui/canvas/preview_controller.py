"""Preview controller for animating rendered cells on the canvas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem

from ...assets import CellTemplate
from ...core.feather import FeatherParameters
from ...core.noise import add_quantization_noise
from ...core.renderer import CellGroundTruth, GroundTruth
from .pixmap_utils import template_pixmap_with_offset


@dataclass(frozen=True)
class _CellPreview:
    """Render-time payload describing a cell animation."""

    item: QGraphicsPixmapItem
    positions: np.ndarray
    angles_rad: np.ndarray
    template: CellTemplate
    offset: Tuple[float, float]


class PreviewController(QObject):
    """Animate simulation previews directly on the main plot canvas."""

    finished = Signal()
    frame_advanced = Signal(int, float)  # (frame_index, time_seconds)

    def __init__(self, plot_item: pg.PlotItem, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._plot_item = plot_item
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._background_item: pg.ImageItem | None = None
        self._background_base: np.ndarray | None = None  # Store original background without noise
        self._background_provider: object | None = None
        self._cells: List[_CellPreview] = []
        self._frame_index = 0
        self._frame_count = 0
        self._frame_interval_ms = 40
        self._fps = 30.0  # Store FPS for time calculation
        self._active = False
        self._pixmap_cache: Dict[str, Tuple[QPixmap, Tuple[float, float]]] = {}
        self._noise_enabled = False
        self._noise_stddev = 5.0
        self._feather_params: Optional[FeatherParameters] = None

    def is_active(self) -> bool:
        """Return whether a preview animation is currently playing."""
        return self._active

    def pause(self) -> None:
        """Pause the preview animation without removing items from canvas."""
        if not self._active:
            return
        self._timer.stop()

    def resume(self) -> None:
        """Resume a paused preview animation."""
        if not self._active:
            return
        self._timer.start(self._frame_interval_ms)

    def seek(self, frame: int) -> None:
        """Seek to a specific frame without starting playback.
        
        Args:
            frame: Target frame number (0-based, clamped to valid range)
        """
        if not self._active:
            return
        
        # Clamp frame to valid range
        frame = max(0, min(frame, self._frame_count - 1))
        self._frame_index = frame
        
        # Update background for this frame (no noise on background)
        if self._background_item is not None:
            bg_frame = None
            if self._background_provider is not None and hasattr(self._background_provider, "frame_for"):
                try:
                    bg_frame = self._background_provider.frame_for(frame)
                except Exception:
                    bg_frame = None
            if bg_frame is None and self._background_base is not None:
                bg_frame = self._background_base
            if bg_frame is not None:
                self._background_item.setImage(bg_frame.T)
                self._background_item.setLevels([0, 255])
        
        # Update cell positions/angles and per-frame noise
        for cell in self._cells:
            if self._noise_enabled:
                pixmap, offset = template_pixmap_with_offset(
                    cell.template,
                    self._feather_params,
                    noise_stddev=self._noise_stddev,
                )
                cell.item.setPixmap(pixmap)
                cell.item.setOffset(offset[0], offset[1])

            if frame < cell.positions.shape[0]:
                pos = cell.positions[frame]
                cell.item.setPos(float(pos[0]), float(pos[1]))
            if frame < cell.angles_rad.shape[0]:
                angle_deg = float(np.degrees(cell.angles_rad[frame]))
                cell.item.setRotation(angle_deg)
        
        # Emit frame update signal
        time_s = self._frame_index / self._fps if self._fps > 0 else 0.0
        self.frame_advanced.emit(self._frame_index, time_s)

    def stop(self) -> None:
        """Stop the preview and remove all preview items from the canvas."""
        if not self._active:
            return
        self._timer.stop()
        self._remove_items()
        # Clear pixmap cache to force regeneration with new scaling parameters
        self._pixmap_cache.clear()
        self._active = False
        self.finished.emit()

    def start(
        self,
        background: np.ndarray,
        templates: Sequence[CellTemplate],
        ground_truth: GroundTruth,
        frame_count: int,
        noise_enabled: bool = False,
        noise_stddev: float = 5.0,
        feather_params: Optional[FeatherParameters] = None,
        background_provider: object | None = None,
    ) -> None:
        """Start animating the preview on the plot canvas."""
        self.stop()

        if frame_count <= 0:
            return

        self._noise_enabled = noise_enabled
        self._noise_stddev = noise_stddev
        self._feather_params = feather_params
        self._background_provider = background_provider
        
        self._prepare_scene(background)
        self._cells = self._build_cells(templates, ground_truth.cells)
        self._frame_index = 0
        self._frame_count = frame_count
        self._fps = ground_truth.fps if ground_truth.fps else 30.0
        self._frame_interval_ms = max(1, int(round(1000.0 / self._fps)))

        if not self._cells:
            # No cells to animate, but we still display the background.
            self._active = True
            self._timer.start(self._frame_interval_ms)
            return

        self._active = True
        self._timer.start(self._frame_interval_ms)

    # Internal helpers -------------------------------------------------

    def _prepare_scene(self, background: np.ndarray) -> None:
        """Clear the plot and add the static background."""
        self._remove_items()
        self._plot_item.clear()
        
        # Store the original background (without noise) for per-frame regeneration
        self._background_base = background.copy()
        
        background_item = pg.ImageItem(background.T)
        background_item.setLevels([0, 255])  # Lock to full uint8 range, no auto-scaling
        background_item.setZValue(0)
        self._plot_item.addItem(background_item)
        self._background_item = background_item

    def _build_cells(
        self,
        templates: Sequence[CellTemplate],
        truths: Iterable[CellGroundTruth],
    ) -> List[_CellPreview]:
        """Create pixmap items for each cell."""
        previews: List[_CellPreview] = []
        for template, cell_truth in zip(templates, truths):
            use_cache = not self._noise_enabled  # dynamic noise when enabled
            cache_key = (
                template.id,
                bool(self._feather_params and self._feather_params.is_active()),
                round(self._noise_stddev, 3),
            )
            cached = self._pixmap_cache.get(cache_key) if use_cache else None
            if cached is None:
                pixmap, offset = template_pixmap_with_offset(
                    template,
                    self._feather_params,
                    noise_stddev=self._noise_stddev if self._noise_enabled else None,
                )
                if use_cache:
                    self._pixmap_cache[cache_key] = (pixmap, offset)
            else:
                pixmap, offset = cached

            item = QGraphicsPixmapItem(pixmap)
            item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            
            # Adjust offset to account for mask centroid position within the template
            item.setOffset(offset[0], offset[1])
            
            # Initialize to frame 0 position BEFORE adding to scene
            # This prevents cells from briefly appearing at (0,0) in top-left corner
            if cell_truth.positions.shape[0] > 0:
                initial_pos = cell_truth.positions[0]
                item.setPos(float(initial_pos[0]), float(initial_pos[1]))
            if cell_truth.cumulative_angle_rad.shape[0] > 0:
                initial_angle_deg = float(np.degrees(cell_truth.cumulative_angle_rad[0]))
                item.setRotation(initial_angle_deg)
            
            item.setZValue(10 + len(previews))
            self._plot_item.vb.addItem(item)
            previews.append(
                _CellPreview(
                    item=item,
                    positions=cell_truth.positions,
                    angles_rad=cell_truth.cumulative_angle_rad,
                    template=template,
                    offset=(offset[0], offset[1]),
                )
            )
        return previews

    def _advance_frame(self) -> None:
        """Advance the animation by one frame."""
        if self._frame_index >= self._frame_count:
            self.stop()
            return

        # Update background for this frame (no noise applied to background)
        if self._background_item is not None:
            bg_frame = None
            if self._background_provider is not None and hasattr(self._background_provider, "frame_for"):
                try:
                    bg_frame = self._background_provider.frame_for(self._frame_index)
                except Exception:
                    bg_frame = None
            if bg_frame is None and self._background_base is not None:
                bg_frame = self._background_base

            if bg_frame is not None:
                self._background_item.setImage(bg_frame.T)
                # Re-lock levels after setImage() to prevent auto-normalization flicker
                self._background_item.setLevels([0, 255])

        # Update cell positions and rotations
        for cell in self._cells:
            # Regenerate noisy pixmap per frame when noise is enabled
            if self._noise_enabled:
                pixmap, offset = template_pixmap_with_offset(
                    cell.template,
                    self._feather_params,
                    noise_stddev=self._noise_stddev,
                )
                cell.item.setPixmap(pixmap)
                cell.item.setOffset(offset[0], offset[1])

            if self._frame_index < cell.positions.shape[0]:
                pos = cell.positions[self._frame_index]
                cell.item.setPos(float(pos[0]), float(pos[1]))
            if self._frame_index < cell.angles_rad.shape[0]:
                angle_deg = float(np.degrees(cell.angles_rad[self._frame_index]))
                cell.item.setRotation(angle_deg)

        # Emit frame update signal
        time_s = self._frame_index / self._fps if self._fps > 0 else 0.0
        self.frame_advanced.emit(self._frame_index, time_s)
        
        self._frame_index += 1

    def _remove_items(self) -> None:
        """Remove preview items from the scene."""
        if self._background_item is not None:
            self._plot_item.removeItem(self._background_item)
            self._background_item = None
        
        # Clear background base for noise regeneration
        self._background_base = None

        for cell in self._cells:
            self._plot_item.vb.removeItem(cell.item)
        self._cells.clear()
