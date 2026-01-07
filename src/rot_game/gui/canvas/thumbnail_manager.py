"""Manages edit-mode thumbnail sprites anchored to scenario trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Set, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
)

from ...assets import CellTemplate, _load_cell_template
from ...config import CellConfig, MaskConfig, TrajectoryConfig
from ...core.feather import FeatherParameters
from ..models import ScenarioCell
from ..utils import calculate_radius_from_mask
from .pixmap_utils import template_pixmap_with_offset


_PixmapPayload = Tuple[QPixmap, QImage, Tuple[float, float]]
_TemplateKey = Tuple[Path, Path, float, float, Tuple, Optional[float]]


def _feather_signature(params: Optional[FeatherParameters]) -> Tuple:
    if not params or not params.is_active():
        return ("off",)
    return (
        "on",
        round(float(params.inside_width_px), 4),
        round(float(params.outside_width_px), 4),
    )


@dataclass
class _SpriteState:
    """Tracks metadata for each on-canvas thumbnail sprite."""

    item: "InteractiveThumbnailItem"
    cell_idx: int
    offset: Tuple[float, float]
    image: Optional[QImage]


class InteractiveThumbnailItem(QGraphicsPixmapItem):
    """Pixmap sprite that forwards mouse/hover events to the thumbnail manager."""

    def __init__(self, manager: "ThumbnailManager", cell_idx: int):
        super().__init__()
        self._manager = manager
        self._cell_idx = cell_idx
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setShapeMode(QGraphicsPixmapItem.ShapeMode.MaskShape)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self._dragging = False

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        inside = self._manager._thumbnail_hit_test(self._cell_idx, event.pos())
        self._manager._handle_hover(self._cell_idx, inside)
        event.accept()

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        inside = self._manager._thumbnail_hit_test(self._cell_idx, event.pos())
        self._manager._handle_hover(self._cell_idx, inside)
        event.accept()

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # type: ignore[override]
        self._manager._handle_hover(self._cell_idx, False)
        event.accept()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._manager._thumbnail_hit_test(self._cell_idx, event.pos())
            and self._manager._handle_press(self._cell_idx, event)
        ):
            self._dragging = True
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and self._manager._handle_move(self._cell_idx, event):
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        consumed = False
        if self._dragging:
            consumed = self._manager._handle_release(self._cell_idx, event)
            self._dragging = False
        if consumed:
            event.accept()
            return
        event.ignore()


class ThumbnailManager:
    """Owns edit-mode thumbnail sprites layered beneath handles."""

    _BASE_Z = 4.5  # Draw above background but below guide lines/handles
    _OPACITY_SELECTED = 1.00
    _OPACITY_DEFAULT = 0.50
    _OPACITY_HOVERED = 0.85

    def __init__(
        self,
        plot_item: pg.PlotItem,
        *,
        on_press: Optional[Callable[[int, QGraphicsSceneMouseEvent], bool]] = None,
        on_move: Optional[Callable[[int, QGraphicsSceneMouseEvent], bool]] = None,
        on_release: Optional[Callable[[int, QGraphicsSceneMouseEvent], bool]] = None,
    ):
        self._plot_item = plot_item
        self._items: Dict[int, _SpriteState] = {}
        self._pixmap_cache: Dict[_TemplateKey, _PixmapPayload] = {}
        self._radius_cache: Dict[Path, float] = {}
        self._enabled = True
        self._selected_idx: Optional[int] = None
        self._hovered_idx: Optional[int] = None
        self._manual_hover_idx: Optional[int] = None
        self._on_press = on_press
        self._on_move = on_move
        self._on_release = on_release

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all sprites and cached data."""
        for state in self._items.values():
            self._plot_item.vb.removeItem(state.item)
        self._items.clear()
        self._selected_idx = None
        self._hovered_idx = None
        self._manual_hover_idx = None

    def set_enabled(self, enabled: bool) -> None:
        """Toggle thumbnail visibility without destroying the cache."""
        if self._enabled == enabled:
            return
        self._enabled = enabled
        for state in self._items.values():
            state.item.setVisible(enabled)
        if not enabled:
            self._manual_hover_idx = None

    def clear_cache_for_cell(self, cell: ScenarioCell) -> None:
        """Invalidate cached pixmaps/radii associated with a cell."""
        template_path = cell.template.resolve()
        mask_path = cell.mask.resolve()
        for cached_key in list(self._pixmap_cache.keys()):
            tpl, msk, *_ = cached_key
            if tpl == template_path and msk == mask_path:
                self._pixmap_cache.pop(cached_key, None)
        self._radius_cache.pop(mask_path, None)

    def clear_cache(self) -> None:
        self._pixmap_cache.clear()
        self._radius_cache.clear()

    def hit_test_scene_point(self, scene_point: QPointF) -> Optional[int]:
        """Return the cell index of the thumbnail covering the scene point."""
        if not self._enabled:
            return None
        for idx, state in self._items.items():
            local_point = state.item.mapFromScene(scene_point)
            if self._thumbnail_hit_test(idx, local_point):
                return idx
        return None

    def set_manual_hover_idx(self, idx: Optional[int]) -> None:
        if self._manual_hover_idx == idx:
            return
        self._manual_hover_idx = idx
        self._apply_opacity_states()

    def rebuild(
        self,
        cells: Sequence[ScenarioCell],
        selected_idx: Optional[int],
        magnification: float,
        pixel_size_um: float,
        feather_params: Optional[FeatherParameters],
        noise_stddev: float | None = None,
    ) -> None:
        """Ensure there is one sprite per cell with up-to-date geometry."""
        self._selected_idx = selected_idx
        keep_indices: Set[int] = set()

        for idx, cell in enumerate(cells):
            keep_indices.add(idx)
            pixmap, image, offset = self._pixmap_for_cell(
                cell,
                magnification,
                pixel_size_um,
                feather_params,
                noise_stddev,
            )
            state = self._items.get(idx)
            if state is None:
                item = InteractiveThumbnailItem(self, idx)
                item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
                item.setZValue(self._BASE_Z + idx * 0.0005)
                self._plot_item.vb.addItem(item)
                state = _SpriteState(item=item, cell_idx=idx, offset=offset, image=image)
                self._items[idx] = state
            else:
                state.item.setPixmap(pixmap)
                if state.item.scene() is None:
                    state.item.setZValue(self._BASE_Z + idx * 0.0005)
                    self._plot_item.vb.addItem(state.item)
            state.offset = offset
            state.image = image
            state.item.setPixmap(pixmap)
            state.item.setOffset(offset[0], offset[1])
            state.item.setPos(float(cell.start[0]), float(cell.start[1]))
            state.item.setOpacity(self._opacity_for(idx))
            state.item.setVisible(self._enabled)

        # Remove sprites for cells that were deleted
        for idx in list(self._items.keys()):
            if idx not in keep_indices:
                state = self._items.pop(idx)
                self._plot_item.vb.removeItem(state.item)
                if self._manual_hover_idx == idx:
                    self._manual_hover_idx = None

    def update_cell_position(self, cell_idx: int, pos: Tuple[float, float]) -> None:
        """Move the sprite for a cell to a new (x, y) coordinate."""
        state = self._items.get(cell_idx)
        if state:
            state.item.setPos(float(pos[0]), float(pos[1]))

    def set_selection(self, selected_idx: Optional[int]) -> None:
        """Mark which sprite is the primary selected one to adjust opacity."""
        self._selected_idx = selected_idx
        self._apply_opacity_states()

    def set_hovered(self, hovered_idx: Optional[int]) -> None:
        """Optional hover feedback for list/canvas interactions."""
        self._hovered_idx = hovered_idx
        self._apply_opacity_states()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_opacity_states(self) -> None:
        if not self._enabled:
            return
        for idx, state in self._items.items():
            state.item.setOpacity(self._opacity_for(idx))

    def _opacity_for(self, idx: int) -> float:
        if idx == self._selected_idx:
            return self._OPACITY_SELECTED
        if self._manual_hover_idx is not None and idx == self._manual_hover_idx:
            return self._OPACITY_HOVERED
        if idx == self._hovered_idx:
            return self._OPACITY_HOVERED
        return self._OPACITY_DEFAULT

    def _pixmap_for_cell(
        self,
        cell: ScenarioCell,
        magnification: float,
        pixel_size_um: float,
        feather_params: Optional[FeatherParameters],
        noise_stddev: float | None,
    ) -> _PixmapPayload:
        key = self._cache_key(cell, magnification, pixel_size_um, feather_params, noise_stddev)
        cached = self._pixmap_cache.get(key)
        if cached is not None:
            return cached

        template = self._build_template(cell, magnification, pixel_size_um)
        pixmap, offset = template_pixmap_with_offset(
            template,
            feather_params,
            noise_stddev=noise_stddev if noise_stddev and noise_stddev > 0 else None,
        )
        image = pixmap.toImage()
        payload = (pixmap, image, offset)
        self._pixmap_cache[key] = payload
        return payload

    def _cache_key(
        self,
        cell: ScenarioCell,
        magnification: float,
        pixel_size_um: float,
        feather_params: Optional[FeatherParameters],
        noise_stddev: float | None,
    ) -> _TemplateKey:
        template_path = cell.template.resolve()
        mask_path = cell.mask.resolve()
        feather_key = _feather_signature(feather_params)
        noise_key = None
        if noise_stddev and noise_stddev > 0:
            noise_key = round(float(noise_stddev), 4)
        return (
            template_path,
            mask_path,
            round(float(magnification), 4),
            round(float(pixel_size_um), 4),
            feather_key,
            noise_key,
        )

    def _build_template(
        self,
        cell: ScenarioCell,
        magnification: float,
        pixel_size_um: float,
    ) -> CellTemplate:
        mask_path = cell.mask.resolve()
        radius_um = self._radius_cache.get(mask_path)
        if radius_um is None:
            radius_um = calculate_radius_from_mask(mask_path)
            self._radius_cache[mask_path] = radius_um
        
        template_path = cell.template.resolve()

        trajectory_cfg = TrajectoryConfig(
            type=cell.trajectory_type,
            start=tuple(cell.start),
            end=tuple(cell.end),
            control_points=[tuple(cp) for cp in cell.control_points] if cell.control_points else None,
            params=getattr(cell, 'params', None) or None,
        )

        cell_cfg = CellConfig(
            id=cell.id,
            radius_um=radius_um,
            template=template_path,
            mask=MaskConfig(path=mask_path),
            trajectory=trajectory_cfg,
        )

        return _load_cell_template(cell_cfg, magnification, pixel_size_um)

    # ------------------------------------------------------------------
    # Interaction plumbing
    # ------------------------------------------------------------------
    def _thumbnail_hit_test(self, cell_idx: int, point) -> bool:
        if not self._enabled:
            return False
        state = self._items.get(cell_idx)
        if state is None or state.image is None:
            return False
        local_x = point.x() - state.offset[0]
        local_y = point.y() - state.offset[1]
        width = state.image.width()
        height = state.image.height()
        if local_x < 0 or local_y < 0 or local_x >= width or local_y >= height:
            return False
        ix = int(local_x)
        iy = int(local_y)
        return state.image.pixelColor(ix, iy).alpha() > 0

    def _handle_hover(self, cell_idx: int, is_inside: bool) -> None:
        if not self._enabled:
            return
        if is_inside:
            self._manual_hover_idx = cell_idx
        else:
            if self._manual_hover_idx == cell_idx:
                self._manual_hover_idx = None
        self._apply_opacity_states()

    def _handle_press(self, cell_idx: int, event: QGraphicsSceneMouseEvent) -> bool:
        if not self._enabled or self._on_press is None:
            return False
        return self._on_press(cell_idx, event)

    def _handle_move(self, cell_idx: int, event: QGraphicsSceneMouseEvent) -> bool:
        if not self._enabled or self._on_move is None:
            return False
        return self._on_move(cell_idx, event)

    def _handle_release(self, cell_idx: int, event: QGraphicsSceneMouseEvent) -> bool:
        if not self._enabled or self._on_release is None:
            return False
        return self._on_release(cell_idx, event)
