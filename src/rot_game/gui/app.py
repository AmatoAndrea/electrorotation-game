"""PySide6/PyQtGraph GUI for simulator authoring."""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, QSize, QPointF, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QAction, QIcon, QColor, QKeySequence, QPainter, QPixmap, QShortcut, QPalette, QImage
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QInputDialog,
    QDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QSlider,
    QStyledItemDelegate,
    QSplitter,
    QTextEdit,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QTabWidget,
    QDoubleSpinBox,
    QHeaderView,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .. import (
    AssetManager,
    CellTemplate,
    GroundTruth,
    Renderer,
    build_frame_schedule,
    export_simulation_outputs,
    load_simulation_config,
)
from ..core.feather import FeatherParameters
from ..catalog import IMAGE_GLOBS, find_mask_for_template, list_cell_lines, list_cell_templates
from ..config import (
    CellConfig,
    FrequencyInterval,
    MaskConfig,
    ScheduleConfig,
    SimulationConfig,
    TrajectoryConfig,
    VideoConfig,
)
from ..settings import asset_root, output_root as default_output_root
from .canvas import HandleManager, PreviewController, TrajectoryRenderer
from .canvas.thumbnail_manager import ThumbnailManager
from .controllers import AssetController, CellController, ConfigController, ScheduleController
from .models import (
    ScenarioCell,
    ScenarioState,
    ScheduleInterval,
    SelectionSnapshot,
    SelectionState,
)
from .utils import frequency_brush, calculate_radius_from_mask
from .widgets import (
    AssetBrowserWidget,
    CellInspectorWidget,
    FloatItemDelegate,
    MaskEditorDialog,
    ScheduleTableWidget,
    ScheduleTimelineWidget,
)


HOVER_PIXEL_TOLERANCE = 5.0
EDGE_FEATHER_SLIDER_SCALE = 10  # tenths of a pixel for slider precision
HANDLE_MIN_DATA_SPAN = 1e-3  # Prevent degenerate ROI sizes when view bounds collapse


logger = logging.getLogger(__name__)


class PanelOverlay(QWidget):
    """Semi-transparent overlay to disable panel interaction during preview mode.
    
    This overlay provides clear visual feedback that editing is locked during preview,
    using a semi-transparent gray background with a centered lock icon and message.
    The overlay excludes the tab bar so users can still switch between tabs.
    """
    
    def __init__(self, parent: QWidget, stop_callback: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        
        # Enable styled background and capture mouse events
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        # Semi-transparent dark overlay (31% opacity)
        self.setStyleSheet("""
            PanelOverlay {
                background-color: rgba(0, 0, 0, 80);
            }
        """)
        
        # Show "not allowed" cursor when hovering
        self.setCursor(Qt.CursorShape.ForbiddenCursor)
        
        # Centered message layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Lock icon and message
        message_row = QWidget(self)
        message_row_layout = QHBoxLayout(message_row)
        message_row_layout.setContentsMargins(0, 0, 0, 0)
        message_row_layout.setSpacing(6)

        icon_label = QLabel(message_row)
        icon_pixmap = self._load_lock_icon(QSize(26, 26))
        if icon_pixmap:
            icon_label.setPixmap(icon_pixmap)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message_row_layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignCenter)

        message_label = QLabel("Preview Mode", message_row)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 220);
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                padding: 0px;
            }
        """)
        message_row_layout.addWidget(message_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        hint_label = QLabel("Stop preview to edit")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 180);
                font-size: 12px;
                background: transparent;
                padding: 4px;
            }
        """)
        
        layout.addStretch()
        layout.addWidget(message_row, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint_label)
        layout.addStretch()
        self._stop_callback = stop_callback

        # Hidden by default
        self.hide()
        
        # Animation for smooth fade-in
        self._fade_animation: Optional[QPropertyAnimation] = None
        
        # Install event filter on parent to catch resize events
        if parent:
            parent.installEventFilter(self)
    
    def eventFilter(self, watched, event):
        """Catch parent resize events to update overlay geometry."""
        if watched == self.parent() and event.type() == QEvent.Type.Resize:
            if self.isVisible():
                self._update_geometry()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:
        """Clicking the overlay stops preview mode."""
        if event.button() == Qt.MouseButton.LeftButton and callable(self._stop_callback):
            self._stop_callback()
            event.accept()
            return
        super().mousePressEvent(event)
    
    def _update_geometry(self):
        """Update overlay geometry to cover content area but exclude tab bar."""
        if not self.parent():
            return
        
        parent_rect = self.parent().rect()
        
        # If parent is QTabWidget, position overlay below tab bar
        if isinstance(self.parent(), QTabWidget):
            tab_widget = self.parent()
            tab_bar = tab_widget.tabBar()
            tab_bar_height = tab_bar.height() if tab_bar else 0
            
            # Position overlay below tab bar, covering only content area
            self.setGeometry(
                0,
                tab_bar_height,
                parent_rect.width(),
                parent_rect.height() - tab_bar_height
            )
        else:
            # For non-tab widgets, cover entire parent
            self.setGeometry(parent_rect)
    
    def showEvent(self, event):
        """Update geometry and animate fade-in when shown."""
        self._update_geometry()
        
        # Smooth fade-in animation
        self.setWindowOpacity(0.0)
        if self._fade_animation:
            self._fade_animation.stop()
        
        self._fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self._fade_animation.setDuration(200)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_animation.start()
        
        super().showEvent(event)
    
    def resizeEvent(self, event):
        """Keep overlay sized correctly when widget resizes."""
        if self.parent() and self.isVisible():
            self._update_geometry()
        super().resizeEvent(event)

    def _load_lock_icon(self, size: QSize) -> Optional[QPixmap]:
        """Render the SVG lock icon to a pixmap for the overlay."""
        icon_path = Path(__file__).resolve().parent / "icons" / "lock.svg"
        if not icon_path.exists():
            logger.warning("Lock icon asset not found: %s", icon_path)
            return None

        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)

        renderer = QSvgRenderer(str(icon_path))
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return pixmap



class TrajectoryItem(pg.PlotCurveItem):
    """Active trajectory curve that supports hover highlighting and dragging."""

    _HOVER_PIXEL_TOLERANCE = HOVER_PIXEL_TOLERANCE

    def __init__(
        self,
        cell_index: int,
        editor,
        selection_state: "SelectionState",
        x_points,
        y_points,
        *args,
        **kwargs,
    ):
        pen = kwargs.pop("pen", pg.mkPen(color=(255, 255, 0), width=3))
        super().__init__(x_points, y_points, pen=pen, *args, **kwargs)
        self.cell_index = cell_index
        self.editor = editor
        self.cell = editor.state.cells[cell_index]
        self._drag_start_scene_pos: Optional[QPointF] = None
        self._drag_start_coords: Optional[Dict[str, Any]] = None
        self._dragging = False
        self._is_hovered = False

        self._normal_pen = pg.mkPen(pen)
        hover_width = max(int(self._normal_pen.widthF()) + 1, 4)
        self._hover_pen = pg.mkPen(color=(255, 165, 0), width=hover_width)

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self._store_curve_points(x_points, y_points)

    def setData(self, *args, **kwargs):  # type: ignore[override]
        super().setData(*args, **kwargs)
        if len(args) >= 2:
            self._store_curve_points(args[0], args[1])
        else:
            x_points = kwargs.get("x")
            y_points = kwargs.get("y")
            if x_points is not None and y_points is not None:
                self._store_curve_points(x_points, y_points)

    def hoverEvent(self, event):  # type: ignore[override]
        if event.isExit():
            if hasattr(self.editor, "_release_curve_hover"):
                self.editor._release_curve_hover(self.cell_index)
            if hasattr(self.editor, "_set_thumbnail_hover_override"):
                self.editor._set_thumbnail_hover_override(None)
            self._set_hover_state(False)
            return

        hovering = self._is_point_near_curve(event.scenePos())
        if hovering and hasattr(self.editor, "_claim_curve_hover"):
            hovering = self.editor._claim_curve_hover(self.cell_index, self.zValue())
        if not hovering and hasattr(self.editor, "_release_curve_hover"):
            self.editor._release_curve_hover(self.cell_index)
        self._set_hover_state(hovering)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        if not self._is_point_near_curve(event.scenePos()):
            event.ignore()
            return

        self.editor._auto_stop_preview_for_edit("trajectory drag")
        self._dragging = True
        self._drag_start_scene_pos = event.scenePos()
        self._drag_start_coords = {
            "start": tuple(self.cell.start),
            "end": tuple(self.cell.end),
            "control_points": [tuple(cp) for cp in self.cell.control_points],
        }
        if hasattr(self.editor, "_begin_curve_drag"):
            self.editor._begin_curve_drag(self.cell_index)
        self._notify_editor_state_change()
        event.accept()

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if not self._dragging or self._drag_start_scene_pos is None or self._drag_start_coords is None:
            event.ignore()
            return

        if not self._apply_drag(event.scenePos()):
            event.ignore()
            return

        event.accept()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            event.ignore()
            return

        self._apply_drag(event.scenePos())
        self._dragging = False
        self._drag_start_scene_pos = None
        self._drag_start_coords = None
        self._notify_editor_state_change()
        if hasattr(self.editor, "_end_curve_drag"):
            self.editor._end_curve_drag(self.cell_index)
        event.accept()

    def _store_curve_points(self, x_points, y_points) -> None:
        self._x_points = np.asarray(x_points, dtype=float)
        self._y_points = np.asarray(y_points, dtype=float)

    def _is_point_near_curve(self, scene_pos: QPointF) -> bool:
        if self._is_over_handle(scene_pos):
            return False

        view_box = self.getViewBox()
        if view_box is None or self._x_points.size < 2:
            return False

        data_pos = view_box.mapSceneToView(scene_pos)
        x = data_pos.x()
        y = data_pos.y()

        pixel_size = view_box.viewPixelSize()
        if isinstance(pixel_size, (tuple, list)):
            pixel_dx, pixel_dy = pixel_size
        else:
            pixel_dx = pixel_size.x()
            pixel_dy = pixel_size.y()

        pixel_scale = max(abs(pixel_dx), abs(pixel_dy)) if pixel_dx and pixel_dy else 1.0
        distance_threshold = max(self._HOVER_PIXEL_TOLERANCE * pixel_scale, 1.0)

        return self._distance_to_curve(x, y) <= distance_threshold

    def _distance_to_curve(self, x: float, y: float) -> float:
        if self._x_points.size < 2:
            return float("inf")

        x1 = self._x_points[:-1]
        y1 = self._y_points[:-1]
        x2 = self._x_points[1:]
        y2 = self._y_points[1:]

        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_len_sq = seg_dx**2 + seg_dy**2
        seg_len_sq[seg_len_sq == 0] = 1e-12

        t = ((x - x1) * seg_dx + (y - y1) * seg_dy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        proj_x = x1 + t * seg_dx
        proj_y = y1 + t * seg_dy

        dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
        return float(np.sqrt(dist_sq.min()))

    def _is_over_handle(self, scene_pos: QPointF) -> bool:
        handle_manager = getattr(self.editor, "_handle_manager", None)
        if handle_manager and handle_manager.is_any_handle_active():
            return True

        handles_dict = self.editor._handles.get(self.cell_index, {})
        for handle in handles_dict.values():
            if handle is None:
                continue
            try:
                # Check if point is within the handle's bounding rectangle
                local_pos = handle.mapFromScene(scene_pos)
                # Use boundingRect to check the full bounding box, not just the shape
                bounding_rect = handle.boundingRect()
                if bounding_rect.contains(local_pos):
                    return True
            except Exception:
                continue

        marker_dict = self.editor._handle_markers.get(self.cell_index, {})
        for marker in marker_dict.values():
            if marker is None:
                continue
            try:
                if marker.pointsAt(scene_pos):
                    return True
            except Exception:
                continue

        return False

    def _apply_drag(self, current_scene_pos: QPointF) -> bool:
        view_box = self.getViewBox()
        if view_box is None or self._drag_start_scene_pos is None or self._drag_start_coords is None:
            return False

        start_data = view_box.mapSceneToView(self._drag_start_scene_pos)
        current_data = view_box.mapSceneToView(current_scene_pos)
        delta_x = current_data.x() - start_data.x()
        delta_y = current_data.y() - start_data.y()

        orig = self._drag_start_coords
        self.cell.start = (orig["start"][0] + delta_x, orig["start"][1] + delta_y)
        self.cell.end = (orig["end"][0] + delta_x, orig["end"][1] + delta_y)
        if orig["control_points"]:
            self.cell.control_points = [
                (cp_x + delta_x, cp_y + delta_y) for (cp_x, cp_y) in orig["control_points"]
            ]

        self.editor._sync_selected_cell_geometry(self.cell)
        return True

    def _set_hover_state(self, hovered: bool) -> None:
        if self._is_hovered == hovered:
            return
        self._is_hovered = hovered
        self.setPen(self._hover_pen if hovered else self._normal_pen)
        if not hovered and hasattr(self.editor, "_release_curve_hover"):
            self.editor._release_curve_hover(self.cell_index)
        self._notify_editor_state_change()

    def _notify_editor_state_change(self) -> None:
        if hasattr(self.editor, "_update_inactive_trajectory_states"):
            self.editor._update_inactive_trajectory_states(exclude=self)


class SelectableTrajectory(pg.PlotCurveItem):
    """Trajectory curve for inactive cells supporting hover highlight and drag."""

    _HOVER_PIXEL_TOLERANCE = HOVER_PIXEL_TOLERANCE

    def __init__(
        self,
        cell_index: int,
        editor,
        selection_state: "SelectionState",
        x_points,
        y_points,
        *args,
        **kwargs,
    ):
        pen = kwargs.pop("pen", pg.mkPen(color=(140, 140, 40, 140), width=2))
        super().__init__(x_points, y_points, pen=pen, *args, **kwargs)
        self.cell_index = cell_index
        self.editor = editor
        self.cell = editor.state.cells[cell_index]
        self.selection_state = selection_state
        self._is_hovered = False
        self._dragging = False
        self._drag_start_scene_pos: Optional[QPointF] = None
        self._drag_start_coords: Optional[Dict[str, Any]] = None
        self._defer_promotion = False

        self._pen_default = pg.mkPen(color=(140, 140, 40, 140), width=2)
        self._pen_hover = pg.mkPen(color=(200, 200, 60, 220), width=3)
        self._pen_selected = pg.mkPen(color=(220, 220, 0, 220), width=2.5)
        self._pen_primary = pg.mkPen(color=(255, 255, 0), width=3)

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self._store_curve_points(x_points, y_points)

        self.selection_state.changed.connect(self._on_selection_changed)
        self.selection_state.hover_changed.connect(self._on_hover_changed)
        self._apply_style(self.selection_state.snapshot())

    def setData(self, *args, **kwargs):  # type: ignore[override]
        super().setData(*args, **kwargs)
        if len(args) >= 2:
            self._store_curve_points(args[0], args[1])
        else:
            x_points = kwargs.get("x")
            y_points = kwargs.get("y")
            if x_points is not None and y_points is not None:
                self._store_curve_points(x_points, y_points)

    def hoverEvent(self, event):  # type: ignore[override]
        if event.isExit():
            if hasattr(self.editor, "_release_curve_hover"):
                self.editor._release_curve_hover(self.cell_index)
            if hasattr(self.editor, "_set_thumbnail_hover_override"):
                self.editor._set_thumbnail_hover_override(None)
            self._set_hover_state(False)
            if self.selection_state.hover == self.cell_index:
                self.selection_state.set_hover(None)
            return

        if self._should_suppress_interaction():
            self._set_hover_state(False)
            if self.selection_state.hover == self.cell_index:
                self.selection_state.set_hover(None)
            return

        scene_pos = event.scenePos()
        thumbnail_owner = None
        if hasattr(self.editor, "_thumbnail_hover_owner"):
            thumbnail_owner = self.editor._thumbnail_hover_owner(scene_pos)
        if thumbnail_owner == self.cell_index:
            if hasattr(self.editor, "_set_thumbnail_hover_override"):
                self.editor._set_thumbnail_hover_override(self.cell_index)
            if self.selection_state.hover == self.cell_index:
                self.selection_state.set_hover(None)
            if hasattr(self.editor, "_release_curve_hover"):
                self.editor._release_curve_hover(self.cell_index)
            self._set_hover_state(False)
            return
        else:
            if hasattr(self.editor, "_set_thumbnail_hover_override"):
                self.editor._set_thumbnail_hover_override(None)

        hovering = self._is_point_near_curve(scene_pos)
        if hovering and hasattr(self.editor, "_claim_curve_hover"):
            hovering = self.editor._claim_curve_hover(self.cell_index, self.zValue())
        if not hovering and hasattr(self.editor, "_release_curve_hover"):
            self.editor._release_curve_hover(self.cell_index)
        self._set_hover_state(hovering)
        if hovering:
            if self.selection_state.hover != self.cell_index:
                self.selection_state.set_hover(self.cell_index)
        else:
            if self.selection_state.hover == self.cell_index:
                self.selection_state.set_hover(None)

    def mousePressEvent(self, event):  # type: ignore[override]
        if self._should_suppress_interaction():
            event.ignore()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        if not self._is_point_near_curve(event.scenePos()):
            event.ignore()
            return

        if self.editor.preview_controller and self.editor.preview_controller.is_active():
            self.editor._on_canvas_mouse_clicked(event)
            return

        # Set up drag state BEFORE triggering selection change to ensure mouse grab is maintained
        print(f"[SelectableTrajectory] mousePressEvent for cell {self.cell_index} - setting up drag state")
        self._defer_promotion = True
        self._dragging = True
        self._drag_start_scene_pos = event.scenePos()
        self._drag_start_coords = {
            "start": tuple(self.cell.start),
            "end": tuple(self.cell.end),
            "control_points": [tuple(cp) for cp in self.cell.control_points],
        }
        print(f"[SelectableTrajectory] Drag state set: _dragging={self._dragging}, _defer_promotion={self._defer_promotion}")
        
        # Now trigger selection change and begin drag
        self.editor._auto_stop_preview_for_edit("trajectory drag")
        print(f"[SelectableTrajectory] Calling _select_cell_from_canvas...")
        self.editor._select_cell_from_canvas(self.cell_index, event.modifiers())
        print(f"[SelectableTrajectory] After _select_cell_from_canvas, _dragging={self._dragging}")
        
        if hasattr(self.editor, "_begin_curve_drag"):
            self.editor._begin_curve_drag(self.cell_index)
        self._notify_editor_state_change()
        print(f"[SelectableTrajectory] mousePressEvent complete, accepting event")
        event.accept()

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if not self._dragging or self._drag_start_scene_pos is None or self._drag_start_coords is None:
            event.ignore()
            return

        if not self._apply_drag(event.scenePos()):
            event.ignore()
            return

        event.accept()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            event.ignore()
            return

        self._apply_drag(event.scenePos())
        self._dragging = False
        self._drag_start_scene_pos = None
        self._drag_start_coords = None
        self._notify_editor_state_change()
        self._defer_promotion = False
        if hasattr(self.editor, "_end_curve_drag"):
            self.editor._end_curve_drag(self.cell_index)
        if self.selection_state.primary == self.cell_index and hasattr(self.editor, "_recreate_active_curve"):
            self.editor._recreate_active_curve(self.cell_index)
        event.accept()

    def _should_suppress_interaction(self) -> bool:
        if self.selection_state.primary == self.cell_index:
            return False
        active = getattr(self.editor, "_active_curve", None)
        if active is not None:
            if getattr(active, "_dragging", False):
                return True

        handle_manager = getattr(self.editor, "_handle_manager", None)
        if handle_manager and handle_manager.is_any_handle_active():
            return True

        if getattr(self.editor, "_is_mouse_near_selected_handles", None):
            if self.editor._is_mouse_near_selected_handles():
                return True

        return False

    def _set_hover_state(self, hovered: bool) -> None:
        if self._is_hovered == hovered:
            return
        self._is_hovered = hovered
        self._apply_style(self.selection_state.snapshot())
        if not hovered and hasattr(self.editor, "_release_curve_hover"):
            self.editor._release_curve_hover(self.cell_index)
        self._notify_editor_state_change()

    def _notify_editor_state_change(self) -> None:
        if hasattr(self.editor, "_update_inactive_trajectory_states"):
            self.editor._update_inactive_trajectory_states(exclude=self)

    def _store_curve_points(self, x_points, y_points) -> None:
        self._x_points = np.asarray(x_points, dtype=float)
        self._y_points = np.asarray(y_points, dtype=float)

    def _is_point_near_curve(self, scene_pos: QPointF) -> bool:
        if self._is_over_handle(scene_pos):
            return False

        view_box = self.getViewBox()
        if view_box is None or self._x_points.size < 2:
            return False

        data_pos = view_box.mapSceneToView(scene_pos)
        x = data_pos.x()
        y = data_pos.y()

        pixel_size = view_box.viewPixelSize()
        if isinstance(pixel_size, (tuple, list)):
            pixel_dx, pixel_dy = pixel_size
        else:
            pixel_dx = pixel_size.x()
            pixel_dy = pixel_size.y()

        pixel_scale = max(abs(pixel_dx), abs(pixel_dy)) if pixel_dx and pixel_dy else 1.0
        distance_threshold = max(self._HOVER_PIXEL_TOLERANCE * pixel_scale, 1.0)

        return self._distance_to_curve(x, y) <= distance_threshold

    def _distance_to_curve(self, x: float, y: float) -> float:
        if self._x_points.size < 2:
            return float("inf")

        x1 = self._x_points[:-1]
        y1 = self._y_points[:-1]
        x2 = self._x_points[1:]
        y2 = self._y_points[1:]

        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_len_sq = seg_dx**2 + seg_dy**2
        seg_len_sq[seg_len_sq == 0] = 1e-12

        t = ((x - x1) * seg_dx + (y - y1) * seg_dy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        proj_x = x1 + t * seg_dx
        proj_y = y1 + t * seg_dy

        dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
        return float(np.sqrt(dist_sq.min()))

    def _is_over_handle(self, scene_pos: QPointF) -> bool:
        handle_manager = getattr(self.editor, "_handle_manager", None)
        if handle_manager and handle_manager.is_any_handle_active():
            return True

        handles_dict = self.editor._handles.get(self.cell_index, {})
        for handle in handles_dict.values():
            if handle is None:
                continue
            try:
                local_pos = handle.mapFromScene(scene_pos)
                bounding_rect = handle.boundingRect()
                if bounding_rect.contains(local_pos):
                    return True
            except Exception:
                continue

        marker_dict = self.editor._handle_markers.get(self.cell_index, {})
        for marker in marker_dict.values():
            if marker is None:
                continue
            try:
                if marker.pointsAt(scene_pos):
                    return True
            except Exception:
                continue

        return False

    def _apply_drag(self, current_scene_pos: QPointF) -> bool:
        view_box = self.getViewBox()
        if view_box is None or self._drag_start_scene_pos is None or self._drag_start_coords is None:
            return False

        start_data = view_box.mapSceneToView(self._drag_start_scene_pos)
        current_data = view_box.mapSceneToView(current_scene_pos)
        delta_x = current_data.x() - start_data.x()
        delta_y = current_data.y() - start_data.y()

        orig = self._drag_start_coords
        self.cell.start = (orig["start"][0] + delta_x, orig["start"][1] + delta_y)
        self.cell.end = (orig["end"][0] + delta_x, orig["end"][1] + delta_y)
        if orig["control_points"]:
            self.cell.control_points = [
                (cp_x + delta_x, cp_y + delta_y) for (cp_x, cp_y) in orig["control_points"]
            ]

        if self.selection_state.primary == self.cell_index:
            self.editor._sync_selected_cell_geometry(self.cell)
        else:
            self.editor._sync_inactive_cell_geometry(self.cell_index, self.cell)
        return True

    def _apply_style(self, snapshot: SelectionSnapshot) -> None:
        if snapshot.primary == self.cell_index:
            self.setPen(self._pen_primary)
            self.setZValue(30)
        elif self.cell_index in snapshot.selected:
            self.setPen(self._pen_selected)
            self.setZValue(20)
        elif self._is_hovered:
            self.setPen(self._pen_hover)
            self.setZValue(15)
        else:
            self.setPen(self._pen_default)
            self.setZValue(10)

    def _on_selection_changed(self, snapshot: SelectionSnapshot) -> None:
        self._apply_style(snapshot)

    def _on_hover_changed(self, hovered_idx: Optional[int]) -> None:
        if hovered_idx != self.cell_index and self._is_hovered:
            self._set_hover_state(False)


class SimulatorEditor(QMainWindow):
    def __init__(self, initial_config: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Electrorotation Game — Scenario Editor")
        self.resize(1500, 860)

        self.state = ScenarioState()
        self.selection_state = SelectionState(self)
        self.selection_state.changed.connect(self._on_selection_changed)
        self.selection_state.hover_changed.connect(self._on_selection_hover_changed)
        self._edge_feather_ui_updating = False
        
        # Initialize controllers
        self.asset_controller = AssetController(self)
        self.cell_controller = CellController(self, self.state)
        self.schedule_controller = ScheduleController(self, self.state)
        self.config_controller = ConfigController(self)
        
        self.thumbnail_icons: Dict[Path, QIcon] = {}
        self.thumbnail_sizes: Dict[Path, Tuple[int, int]] = {}
        self._background_paths: List[Path] = []
        self._handle_manager: Optional[HandleManager] = None  # Created after plot_item
        self._thumbnail_manager: Optional[ThumbnailManager] = None
        self._inactive_trajectories: Dict[int, SelectableTrajectory] = {}
        self._last_scene_pos: Optional[QPointF] = None
        self._curve_drag_stack = 0
        self._handles_disabled: Set[int] = set()
        # Keep these for backward compatibility with existing code
        self._handles: Dict[int, Dict[str, pg.ROI]] = {}
        self._handle_markers: Dict[int, Dict[str, pg.ScatterPlotItem]] = {}
        self._guide_lines: Dict[int, List[pg.PlotDataItem]] = {}
        self._active_curve: Optional[pg.PlotCurveItem] = None
        self._trajectory_items: Dict[int, TrajectoryItem] = {}
        self._inactive_markers: Dict[int, List[pg.ScatterPlotItem]] = {}
        self._current_primary: Optional[int] = None
        self._background_extent: Optional[Tuple[int, int]] = None
        self._shortcuts: List[QShortcut] = []
        self.preview_controller: Optional[PreviewController] = None
        self._preview_finish_message: Optional[str] = None
        self.preview_button: Optional[QActionButton] = None
        self._thumbnail_drag_state: Optional[Dict[str, Any]] = None
        self._thumbnail_hover_override: Optional[int] = None
        self._hover_owner: Optional[int] = None
        self._hover_owner_z: float = float("-inf")
        self._handle_viewbox_connected = False
        
        # Icon assets
        self._icon_dir = Path(__file__).resolve().parent / "icons"
        self._icon_cache: Dict[str, QIcon] = {}
        self._examples_dir = self._resolve_examples_dir()

        # Empty state widget to replace canvas when no background
        self._welcome_widget: Optional[QWidget] = None

        self._setup_ui()
        
        # Initialize control states based on empty scenario
        self._update_control_states()
        
        if initial_config:
            self.load_config(initial_config)

    def _set_scrollbar_thickness(self, widget: Any, thickness: int = 8) -> None:
        """Force a consistent scrollbar thickness without restyling handles."""
        get_v = getattr(widget, "verticalScrollBar", None)
        if callable(get_v):
            bar = get_v()
            if bar:
                bar.setFixedWidth(thickness)
        get_h = getattr(widget, "horizontalScrollBar", None)
        if callable(get_h):
            bar = get_h()
            if bar:
                bar.setFixedHeight(thickness)

    def _resolve_examples_dir(self) -> Path:
        """Return simulator/examples directory if available, else asset root."""
        examples_dir = Path(__file__).resolve().parents[3] / "examples"
        if examples_dir.exists():
            return examples_dir
        return asset_root()

    def _setup_ui(self) -> None:
        self._create_file_actions()
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Build constituent panels
        sidebar = self._build_left_panel()
        center_panel = self._build_center_panel()

        # Main splitter (left sidebar, right canvas/timeline)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(sidebar)
        self.main_splitter.addWidget(center_panel)
        self.main_splitter.setSizes([400, 1100])

        main_layout.addWidget(self.main_splitter)

        self.statusBar().showMessage("Ready")
        self.statusBar().setVisible(False)
        self.show_status_bar_action = QAction("Show Status Bar", self)
        self.show_status_bar_action.setCheckable(True)
        self.show_status_bar_action.setChecked(False)
        self.show_status_bar_action.toggled.connect(self.statusBar().setVisible)
        self._register_shortcuts()
        self._build_menu_bar()

    def _create_file_actions(self) -> None:
        """Instantiate reusable actions shared between menus and shortcuts."""
        self.open_config_action = QAction("Open Config...", self)
        self.open_config_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_config_action.triggered.connect(self.open_config_dialog)
        self._apply_action_icon(self.open_config_action, "folder_open")
        self.addAction(self.open_config_action)

        self.save_config_action = QAction("Save Config", self)
        self.save_config_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_config_action.triggered.connect(self.save_config_dialog)
        self._apply_action_icon(self.save_config_action, "save_as")
        self.addAction(self.save_config_action)

        self.save_video_action = QAction("Save Video...", self)
        self.save_video_action.triggered.connect(self.render_current_scenario)
        self._apply_action_icon(self.save_video_action, "movie_edit")
        self.addAction(self.save_video_action)

    def _build_menu_bar(self) -> None:
        """Populate the native/global menu bar with existing actions."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.open_config_action)
        file_menu.addAction(self.save_config_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_video_action)
        file_menu.addSeparator()
        self.quit_action = QAction("Quit", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(QApplication.instance().quit)  # type: ignore[attr-defined]
        file_menu.addAction(self.quit_action)

        view_menu = menu_bar.addMenu("View")
        if hasattr(self, "fit_view_action"):
            view_menu.addAction(self.fit_view_action)
        if hasattr(self, "zoom_in_action"):
            view_menu.addAction(self.zoom_in_action)
        if hasattr(self, "zoom_out_action"):
            view_menu.addAction(self.zoom_out_action)
        if hasattr(self, "show_thumbnails_action"):
            view_menu.addAction(self.show_thumbnails_action)
        if hasattr(self, "show_status_bar_action"):
            view_menu.addAction(self.show_status_bar_action)

        playback_menu = menu_bar.addMenu("Playback")
        if hasattr(self, "play_pause_action"):
            playback_menu.addAction(self.play_pause_action)
        if hasattr(self, "stop_action"):
            playback_menu.addAction(self.stop_action)

        help_menu = menu_bar.addMenu("Help")
        about_qt_action = QAction("About Qt", self)
        about_qt_action.triggered.connect(QApplication.instance().aboutQt)  # type: ignore[attr-defined]
        help_menu.addAction(about_qt_action)

    # Left panel ------------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.sidebar_tabs.setDocumentMode(True)
        self.sidebar_tabs.setMinimumWidth(340)

        # Assets tab (backgrounds + cells + inspector) wrapped in scroll area
        assets_tab = QWidget()
        assets_layout = QVBoxLayout(assets_tab)
        assets_layout.setContentsMargins(0, 0, 0, 0)
        assets_layout.setSpacing(0)

        assets_content = QWidget()
        assets_content_layout = QVBoxLayout(assets_content)
        assets_content_layout.setContentsMargins(8, 8, 8, 8)
        assets_content_layout.setSpacing(12)

        self.asset_browser = AssetBrowserWidget()
        assets_content_layout.addWidget(self.asset_browser)
        self._set_scrollbar_thickness(self.asset_browser.cell_template_list)

        self.cell_inspector = CellInspectorWidget()
        self.cell_inspector.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        assets_content_layout.addWidget(self.cell_inspector)
        self._set_scrollbar_thickness(self.cell_inspector.scenario_cell_list)
        assets_content_layout.addStretch(1)

        assets_scroll = QScrollArea()
        assets_scroll.setWidgetResizable(True)
        assets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        assets_scroll.setWidget(assets_content)
        self._set_scrollbar_thickness(assets_scroll)

        assets_layout.addWidget(assets_scroll)
        assets_index = self.sidebar_tabs.addTab(assets_tab, "Assets")
        self._set_tab_icon(assets_index, "shape_line")

        # Schedule tab (reuse existing widget)
        schedule_tab = QWidget()
        schedule_layout = QVBoxLayout(schedule_tab)
        schedule_layout.setContentsMargins(0, 8, 0, 8)
        schedule_layout.setSpacing(0)

        self.schedule_widget = ScheduleTableWidget()
        self.schedule_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        schedule_layout.addWidget(self.schedule_widget)

        schedule_index = self.sidebar_tabs.addTab(schedule_tab, "Schedule")
        self._set_tab_icon(schedule_index, "table")

        # Video Settings tab (new)
        video_settings_tab = QWidget()
        video_settings_layout = QVBoxLayout(video_settings_tab)
        video_settings_layout.setContentsMargins(8, 8, 8, 8)
        video_settings_layout.setSpacing(12)

        # Video Settings Group
        video_group = QGroupBox("Video Configuration")
        video_group_layout = QVBoxLayout(video_group)
        video_group_layout.setContentsMargins(8, 8, 8, 8)
        video_group_layout.setSpacing(8)
        video_form = QFormLayout()
        video_form.setSpacing(8)

        # Resolution (width x height)
        resolution_layout = QHBoxLayout()
        self.resolution_width_spin = QDoubleSpinBox()
        self.resolution_width_spin.setRange(320, 3840)
        self.resolution_width_spin.setDecimals(0)
        self.resolution_width_spin.setValue(640)
        self.resolution_width_spin.setSuffix(" px")
        self.resolution_width_spin.valueChanged.connect(self._on_resolution_changed)
        
        resolution_x_label = QLabel("×")
        
        self.resolution_height_spin = QDoubleSpinBox()
        self.resolution_height_spin.setRange(240, 2160)
        self.resolution_height_spin.setDecimals(0)
        self.resolution_height_spin.setValue(480)
        self.resolution_height_spin.setSuffix(" px")
        self.resolution_height_spin.valueChanged.connect(self._on_resolution_changed)
        
        resolution_layout.addWidget(self.resolution_width_spin)
        resolution_layout.addWidget(resolution_x_label)
        resolution_layout.addWidget(self.resolution_height_spin)
        resolution_layout.addStretch()
        video_form.addRow("Resolution:", resolution_layout)

        # FPS
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 240.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        video_form.addRow("Frame Rate:", self.fps_spin)
        video_group_layout.addLayout(video_form)

        self.video_summary_label = QLabel()
        self.video_summary_label.setWordWrap(True)
        self.video_summary_label.setStyleSheet("color: #888; font-size: 10pt; padding-left: 4px;")
        video_group_layout.addWidget(self.video_summary_label)

        video_settings_layout.addWidget(video_group)

        # Microscopy Settings Group
        microscopy_group = QGroupBox("Microscopy Parameters")
        microscopy_layout = QVBoxLayout(microscopy_group)
        microscopy_layout.setContentsMargins(8, 8, 8, 8)
        microscopy_layout.setSpacing(8)
        microscopy_form = QFormLayout()
        microscopy_form.setSpacing(8)

        # Background Reference Magnification (first)
        self.bg_ref_mag_spin = QDoubleSpinBox()
        self.bg_ref_mag_spin.setRange(1.0, 150.0)
        self.bg_ref_mag_spin.setDecimals(2)
        self.bg_ref_mag_spin.setValue(10.93)
        self.bg_ref_mag_spin.setSuffix(" ×")
        self.bg_ref_mag_spin.valueChanged.connect(self._on_bg_ref_mag_changed)
        microscopy_form.addRow("Background Ref. Mag.:", self.bg_ref_mag_spin)

        # Magnification (second)
        self.magnification_spin = QDoubleSpinBox()
        self.magnification_spin.setRange(1.0, 150.0)
        self.magnification_spin.setDecimals(2)
        self.magnification_spin.setValue(10.93)
        self.magnification_spin.setSuffix(" ×")
        self.magnification_spin.valueChanged.connect(self._on_magnification_changed)
        microscopy_form.addRow("Magnification:", self.magnification_spin)
        
        # Warning label for magnification (appears when mag < bg_ref_mag)
        self.mag_warning_label = QLabel("⚠️ Magnification below reference - background padding applied")
        self.mag_warning_label.setStyleSheet("color: #FFA500; font-size: 10pt; padding-left: 4px;")
        self.mag_warning_label.setWordWrap(True)  # Enable word wrap to prevent truncation
        self.mag_warning_label.setVisible(False)
        microscopy_form.addRow("", self.mag_warning_label)

        # Pixel Size (third)
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.1, 50.0)
        self.pixel_size_spin.setDecimals(2)
        self.pixel_size_spin.setValue(7.4)
        self.pixel_size_spin.setSuffix(" µm/px")
        self.pixel_size_spin.valueChanged.connect(self._on_pixel_size_changed)
        microscopy_form.addRow("Pixel Size:", self.pixel_size_spin)
        microscopy_layout.addLayout(microscopy_form)

        self.effective_pixel_label = QLabel()
        self.effective_pixel_label.setWordWrap(True)
        self.effective_pixel_label.setStyleSheet("color: #888; font-size: 10pt; padding-left: 4px;")
        microscopy_layout.addWidget(self.effective_pixel_label)

        video_settings_layout.addWidget(microscopy_group)

        # Noise simulation group
        noise_group = QGroupBox("Quantization Noise")
        noise_layout = QVBoxLayout()
        noise_group.setLayout(noise_layout)
        
        noise_form = QFormLayout()
        noise_form.setSpacing(8)
        
        # Noise enabled button
        self.noise_enabled_check = QPushButton("Enable Noise")
        self.noise_enabled_check.setCheckable(True)
        self.noise_enabled_check.setChecked(False)
        self.noise_enabled_check.toggled.connect(self._on_noise_enabled_changed)
        noise_form.addRow("", self.noise_enabled_check)
        
        # Noise level slider
        self.noise_stddev_spin = QDoubleSpinBox()
        self.noise_stddev_spin.setRange(0.5, 50.0)
        self.noise_stddev_spin.setDecimals(1)
        self.noise_stddev_spin.setSingleStep(0.5)
        self.noise_stddev_spin.setValue(5.0)
        self.noise_stddev_spin.setSuffix(" σ")
        self.noise_stddev_spin.valueChanged.connect(self._on_noise_stddev_changed)
        # Always enabled - can adjust level even when noise is off
        noise_form.addRow("Noise Level:", self.noise_stddev_spin)
        
        noise_layout.addLayout(noise_form)
        
        # Noise info label
        noise_info = QLabel("Simulates camera read noise.\nTypical: 2-5 (subtle), 5-10 (moderate), 10-20 (heavy), 20-50 (extreme)")
        noise_info.setWordWrap(True)
        noise_info.setStyleSheet("color: #888; font-size: 10pt; padding: 4px;")
        noise_layout.addWidget(noise_info)
        
        video_settings_layout.addWidget(noise_group)

        # Edge Feathering group (always visible for quick access)
        self.edge_feather_group = self._build_edge_feathering_group()
        video_settings_layout.addWidget(self.edge_feather_group)

        self._update_video_info_label()
        self._sync_edge_feather_controls_from_state()

        video_settings_layout.addStretch()

        video_index = self.sidebar_tabs.addTab(video_settings_tab, "Video")
        self._set_tab_icon(video_index, "video_settings")

        # Connect asset browser signals to handlers
        self.asset_browser.background_folder_clicked.connect(self._on_background_folder_clicked)
        self.asset_browser.cell_line_changed.connect(self._on_cell_line_changed)
        self.asset_browser.template_double_clicked.connect(self._on_cell_template_double_clicked)
        self.asset_browser.template_context_menu_requested.connect(self._on_template_context_menu)
        self.asset_browser.add_template_requested.connect(self._on_add_template_requested)

        # Connect inspector signals
        self.cell_inspector.cell_selected.connect(self._on_scenario_cell_selected)
        self.cell_inspector.remove_cells_clicked.connect(self._remove_selected_cells)
        self.cell_inspector.scenario_context_menu_requested.connect(self._on_scenario_context_menu)
        self.cell_inspector.trajectory_type_changed.connect(self._on_trajectory_type_changed)
        self.cell_inspector.coordinate_changed.connect(self._on_position_component_changed)
        self.cell_inspector.control_point_changed.connect(self._on_control_component_changed)
        self.cell_inspector.trajectory_param_changed.connect(self._on_trajectory_param_changed)
        self.cell_inspector.edit_mask_requested.connect(self._open_mask_editor_for_selection)
        self.cell_inspector.rename_cell_requested.connect(self._rename_scenario_cell)

        # Connect schedule signals
        self.schedule_widget.add_interval_clicked.connect(self._add_schedule_interval)
        self.schedule_widget.remove_interval_clicked.connect(self._remove_schedule_interval)
        self.schedule_widget.load_schedule_clicked.connect(self._load_schedule_from_yaml)
        self.schedule_widget.context_menu_requested.connect(self._on_schedule_context_menu)
        self.schedule_widget.install_event_filter(self)
        self._schedule_table_refreshing = False
        self.schedule_table = self.schedule_widget.schedule_table
        self.schedule_table.itemChanged.connect(self._on_schedule_item_changed)
        self._set_scrollbar_thickness(self.schedule_table)

        # Expose widget components for backward compatibility
        self.cell_line_combo = self.asset_browser.cell_line_combo
        self.cell_template_list = self.asset_browser.cell_template_list
        self.scenario_cell_list = self.cell_inspector.scenario_cell_list
        self.cell_id_label = self.cell_inspector.cell_id_label
        self.trajectory_combo = self.cell_inspector.trajectory_combo
        self.cell_preview_label = self.cell_inspector.cell_preview_label
        self.cell_preview_edit_button = self.cell_inspector.edit_mask_button
        self.start_label = self.cell_inspector.start_label
        self.start_x_spin = self.cell_inspector.start_x_spin
        self.start_y_spin = self.cell_inspector.start_y_spin
        self.end_label = self.cell_inspector.end_label
        self.end_x_spin = self.cell_inspector.end_x_spin
        self.end_y_spin = self.cell_inspector.end_y_spin
        self.control1_label = self.cell_inspector.control1_label
        self.control1_x_spin = self.cell_inspector.control1_x_spin
        self.control1_y_spin = self.cell_inspector.control1_y_spin
        self.control2_label = self.cell_inspector.control2_label
        self.control2_x_spin = self.cell_inspector.control2_x_spin
        self.control2_y_spin = self.cell_inspector.control2_y_spin
        self.add_interval_button = self.schedule_widget.add_interval_button
        self.remove_interval_button = self.schedule_widget.remove_interval_button

        self._reload_asset_lists()
        
        # Create overlay on sidebar_tabs (covers content area, not tabs)
        self.left_panel_overlay = PanelOverlay(self.sidebar_tabs, stop_callback=self._on_stop_clicked)
        
        return self.sidebar_tabs

    def _register_shortcuts(self) -> None:
        shortcuts: List[Tuple[QKeySequence, Callable[[], None]]] = [
            (QKeySequence("Ctrl+1"), lambda: self.sidebar_tabs.setCurrentIndex(0)),
            (QKeySequence("Ctrl+2"), lambda: self.sidebar_tabs.setCurrentIndex(1)),
            (QKeySequence("Ctrl+3"), lambda: self.sidebar_tabs.setCurrentIndex(2)),
        ]
        cell_shortcuts: List[Tuple[QKeySequence, Callable[[], None]]] = [
            (QKeySequence(Qt.Key.Key_Delete), self._handle_remove_cell_shortcut),
            (QKeySequence(Qt.Key.Key_Backspace), self._handle_remove_cell_shortcut),
            (QKeySequence(Qt.Key.Key_S), lambda: self._handle_trajectory_shortcut("stationary")),
            (QKeySequence(Qt.Key.Key_L), lambda: self._handle_trajectory_shortcut("linear")),
            (QKeySequence(Qt.Key.Key_P), lambda: self._handle_trajectory_shortcut("parabolic")),
            (QKeySequence(Qt.Key.Key_C), lambda: self._handle_trajectory_shortcut("cubic")),
        ]
        self._shortcuts.clear()
        for sequence, handler in [*shortcuts, *cell_shortcuts]:
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)

    def _can_handle_cell_shortcut(self) -> bool:
        """Return True if global cell shortcuts should be honored."""
        focused = QApplication.focusWidget()
        text_like = (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)
        if isinstance(focused, text_like):
            return False
        if getattr(self, "schedule_table", None):
            if focused is self.schedule_table or self.schedule_table.isAncestorOf(focused):
                return False
        return True

    def _handle_remove_cell_shortcut(self) -> None:
        if not self._can_handle_cell_shortcut():
            return
        if not self.state.cells or not self.scenario_cell_list.selectedItems():
            return
        self._remove_selected_cells()

    def _handle_trajectory_shortcut(self, trajectory_type: str) -> None:
        if not self._can_handle_cell_shortcut():
            return
        cell = self._current_cell()
        if cell is None or cell.trajectory_type == trajectory_type:
            return
        self._on_trajectory_type_changed(trajectory_type)
    # Center panel ----------------------------------------------------------

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        canvas_toolbar = QToolBar("Canvas", self)
        canvas_toolbar.setIconSize(QSize(16, 16))
        canvas_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout.addWidget(canvas_toolbar)

        # Preview canvas
        self.plot_widget = pg.GraphicsLayoutWidget()

        self.plot_item = self.plot_widget.addPlot()
        self.plot_item.invertY(True)
        self.plot_item.showGrid(alpha=0.3)
        self.plot_item.setAspectLocked(True, ratio=1.0)
        self._ensure_handle_viewbox_sync()
        
        # Create welcome widget (hidden by default, shown when no background)
        self._welcome_widget = self._create_welcome_widget()
        
        # Container to switch between canvas and welcome screen
        self.canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.plot_widget)
        canvas_layout.addWidget(self._welcome_widget)
        
        # Show welcome initially (no background at startup)
        self.plot_widget.hide()
        self._welcome_widget.show()
        
        # Initialize managers after plot_item is created
        self._handle_manager = HandleManager(self.plot_item)
        self._thumbnail_manager = ThumbnailManager(
            self.plot_item,
            on_press=self._on_thumbnail_mouse_press,
            on_move=self._on_thumbnail_mouse_move,
            on_release=self._on_thumbnail_mouse_release,
        )
        self._thumbnails_enabled = True
        self.preview_controller = PreviewController(self.plot_item)
        self.preview_controller.finished.connect(self._on_preview_finished)
        self.preview_controller.frame_advanced.connect(self._on_preview_frame_advanced)
        scene = self.plot_widget.scene()
        if hasattr(scene, "sigMouseClicked"):
            scene.sigMouseClicked.connect(self._on_canvas_mouse_clicked)  # type: ignore[attr-defined]
        if hasattr(scene, "sigMouseMoved"):
            scene.sigMouseMoved.connect(self._on_canvas_mouse_moved)  # type: ignore[attr-defined]
        
        # interactive handles will manage picking, so no raw click handler here
        layout.addWidget(self.canvas_container, 1)

        self.fit_view_action = QAction("Fit View", self)
        self.fit_view_action.setShortcut(QKeySequence(Qt.Key.Key_F))
        self.fit_view_action.triggered.connect(self._fit_view)
        self._apply_action_icon(self.fit_view_action, "fit_screen")
        canvas_toolbar.addAction(self.fit_view_action)
        self.addAction(self.fit_view_action)

        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.triggered.connect(lambda: self.plot_item.vb.scaleBy((0.8, 0.8)))
        self._apply_action_icon(self.zoom_in_action, "zoom_in")
        canvas_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.triggered.connect(lambda: self.plot_item.vb.scaleBy((1.25, 1.25)))
        self._apply_action_icon(self.zoom_out_action, "zoom_out")
        canvas_toolbar.addAction(self.zoom_out_action)

        self.thumbnail_checkbox = QCheckBox("Show Thumbnails")
        self.thumbnail_checkbox.setChecked(True)
        self.thumbnail_checkbox.toggled.connect(
            lambda checked: self._on_thumbnail_toggle_changed(checked, "checkbox")
        )
        thumb_action = QWidgetAction(self)
        thumb_action.setDefaultWidget(self.thumbnail_checkbox)
        canvas_toolbar.addAction(thumb_action)
        self.show_thumbnails_widget_action = thumb_action

        self.show_thumbnails_action = QAction("Show Thumbnails", self)
        self.show_thumbnails_action.setCheckable(True)
        self.show_thumbnails_action.toggled.connect(
            lambda checked: self._on_thumbnail_toggle_changed(checked, "action")
        )
        self.show_thumbnails_action.setChecked(True)

        # Timeline panel with controls
        timeline_container = QWidget()
        timeline_container.setFixedHeight(150)
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(8, 6, 8, 6)
        timeline_layout.setSpacing(4)
        
        # Timeline widget
        self.schedule_timeline = ScheduleTimelineWidget(self)
        self.schedule_timeline.setFixedHeight(60)
        self.schedule_timeline.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        timeline_layout.addWidget(self.schedule_timeline)
        
        # Initialize timeline with sensible defaults (10 seconds at 30 fps)
        self._initialize_empty_timeline()
        
        # Connect timeline signals
        self.schedule_timeline.seek_requested.connect(self._on_timeline_seek)
        
        # Button bar below timeline
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        
        # Play/Pause button (left aligned)
        self.play_pause_button = QPushButton()
        self._apply_icon(self.play_pause_button, "play")
        self.play_pause_button.setText("Play")
        self.play_pause_button.clicked.connect(self._on_play_pause_clicked)
        button_layout.addWidget(self.play_pause_button)
        
        # Stop button
        self.stop_button = QPushButton()
        self._apply_icon(self.stop_button, "stop")
        self.stop_button.setText("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self.stop_button)

        # Time/Frame readout
        self.timeline_status_label = QLabel("0:00 / 0:00 • Frame 0 / 0")
        self.timeline_status_label.setStyleSheet("color: #bbb; font-size: 11pt;")
        button_layout.addWidget(self.timeline_status_label)
        self._update_timeline_status_label()
        
        self.play_pause_action = QAction("Play Preview", self)
        self.play_pause_action.triggered.connect(self._on_play_pause_clicked)
        self._apply_action_icon(self.play_pause_action, "play")
        self.play_pause_action.setEnabled(False)

        self.stop_action = QAction("Stop Preview", self)
        self.stop_action.triggered.connect(self._on_stop_clicked)
        self._apply_action_icon(self.stop_action, "stop")
        self.stop_action.setEnabled(False)

        self.addAction(self.play_pause_action)
        self.addAction(self.stop_action)
        
        # Spacer to push Save Video to the right
        button_layout.addStretch()
        
        # Save Video button (right aligned)
        self.export_button = QActionButton("Save Video...", self.render_current_scenario)
        self._apply_icon(self.export_button.button, "movie_edit")
        button_layout.addWidget(self.export_button.button)
        
        timeline_layout.addLayout(button_layout)

        layout.addWidget(timeline_container, 0)

        return panel


    def _on_thumbnail_toggle_changed(self, checked: bool, source: str) -> None:
        """Synchronize thumbnail toggle state between toolbar and menu."""
        checkbox = getattr(self, "thumbnail_checkbox", None)
        if checkbox is not None and source != "checkbox":
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

        action = getattr(self, "show_thumbnails_action", None)
        if action is not None and source != "action":
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

        self._toggle_thumbnails(checked)

    def _toggle_thumbnails(self, checked: bool) -> None:
        if not self._thumbnail_manager:
            return
        self._thumbnails_enabled = checked
        self._thumbnail_manager.set_enabled(checked)
        if checked:
            selected_idx = self._current_cell_index()
            if selected_idx < 0 or selected_idx >= len(self.state.cells):
                selected_idx = None
            self._thumbnail_manager.rebuild(
                self.state.cells,
                selected_idx,
                self.state.magnification,
                self.state.pixel_size_um,
                self._edge_feather_runtime_params(),
                self.state.noise_stddev if self.state.noise_enabled else None,
            )


    def _build_edge_feathering_group(self) -> QGroupBox:
        """Create the always-visible Edge Feathering controls for the Video tab."""
        group = QGroupBox("Edge Feathering")
        layout = QFormLayout(group)
        layout.setSpacing(8)

        slider_min = self._edge_feather_slider_min_value()
        slider_max = self._edge_feather_slider_max_value()

        # Enable Feather toggle
        self.edge_feather_enable_button = QPushButton("Enable Feather")
        self.edge_feather_enable_button.setCheckable(True)
        self.edge_feather_enable_button.setChecked(False)
        self.edge_feather_enable_button.toggled.connect(self._on_edge_feather_enable_toggled)
        layout.addRow("Enable feather:", self.edge_feather_enable_button)

        # Inside width controls
        inside_layout = QHBoxLayout()
        inside_layout.setContentsMargins(0, 0, 0, 0)
        inside_layout.setSpacing(6)
        self.edge_feather_inside_spin = QDoubleSpinBox()
        self.edge_feather_inside_spin.setRange(
            self.state.edge_feather.min_pixels,
            self.state.edge_feather.max_pixels,
        )
        self.edge_feather_inside_spin.setDecimals(2)
        self.edge_feather_inside_spin.setValue(self.state.edge_feather.inside_pixels)
        self.edge_feather_inside_spin.setSuffix(" px")
        self.edge_feather_inside_spin.valueChanged.connect(self._on_edge_feather_inside_spin_changed)
        inside_layout.addWidget(self.edge_feather_inside_spin, 0)

        self.edge_feather_inside_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_feather_inside_slider.setRange(slider_min, slider_max)
        self.edge_feather_inside_slider.setSingleStep(1)
        self.edge_feather_inside_slider.setPageStep(5)
        self.edge_feather_inside_slider.setValue(self._px_to_edge_slider(self.state.edge_feather.inside_pixels))
        self.edge_feather_inside_slider.valueChanged.connect(self._on_edge_feather_inside_slider_changed)
        inside_layout.addWidget(self.edge_feather_inside_slider, 1)

        self.edge_feather_inside_um_label = QLabel("≈ — µm")
        self.edge_feather_inside_um_label.setStyleSheet("color: #888;")
        inside_layout.addWidget(self.edge_feather_inside_um_label, 0)

        layout.addRow("Inside width:", inside_layout)

        # Outside width controls
        outside_layout = QHBoxLayout()
        outside_layout.setContentsMargins(0, 0, 0, 0)
        outside_layout.setSpacing(6)
        self.edge_feather_outside_spin = QDoubleSpinBox()
        self.edge_feather_outside_spin.setRange(
            self.state.edge_feather.min_pixels,
            self.state.edge_feather.max_pixels,
        )
        self.edge_feather_outside_spin.setDecimals(2)
        self.edge_feather_outside_spin.setValue(self.state.edge_feather.outside_pixels)
        self.edge_feather_outside_spin.setSuffix(" px")
        self.edge_feather_outside_spin.valueChanged.connect(self._on_edge_feather_outside_spin_changed)
        outside_layout.addWidget(self.edge_feather_outside_spin, 0)

        self.edge_feather_outside_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_feather_outside_slider.setRange(slider_min, slider_max)
        self.edge_feather_outside_slider.setSingleStep(1)
        self.edge_feather_outside_slider.setPageStep(5)
        self.edge_feather_outside_slider.setValue(self._px_to_edge_slider(self.state.edge_feather.outside_pixels))
        self.edge_feather_outside_slider.valueChanged.connect(self._on_edge_feather_outside_slider_changed)
        outside_layout.addWidget(self.edge_feather_outside_slider, 1)

        self.edge_feather_outside_um_label = QLabel("≈ — µm")
        self.edge_feather_outside_um_label.setStyleSheet("color: #888;")
        outside_layout.addWidget(self.edge_feather_outside_um_label, 0)
        layout.addRow("Outside width:", outside_layout)

        # Warning label (shares style with magnification warning)
        self.edge_feather_warning_label = QLabel(
            "⚠️ Provide pixel size and magnification to compute physical widths"
        )
        self.edge_feather_warning_label.setStyleSheet("color: #FFA500; font-size: 10pt; padding-left: 4px;")
        self.edge_feather_warning_label.setWordWrap(True)
        self.edge_feather_warning_label.setVisible(False)
        layout.addRow("", self.edge_feather_warning_label)

        return group

    # Right panel -----------------------------------------------------------

    # Asset handling ------------------------------------------------------

    def _reload_asset_lists(self) -> None:
        # Only refresh templates (background handled via folder picker)
        # Reload cell lines and templates
        self.asset_controller.reload_all_assets(
            background_list=None,
            cell_line_combo=self.cell_line_combo,
            search_text="",
        )
        self.asset_browser.ensure_cell_line_placeholder(getattr(self, "current_line", None))
        if getattr(self, "current_line", None):
            idx = self.cell_line_combo.findText(self.current_line)
            if idx != -1:
                self.cell_line_combo.setCurrentIndex(idx)
                self._populate_templates_for_line(self.current_line)
        else:
            self.cell_template_list.clear()

    def _filter_backgrounds(self, text: Optional[str] = None) -> None:
        # Deprecated: backgrounds are selected via folder picker
        return

    def _on_background_folder_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Background Frame Folder", str(asset_root()))
        if not directory:
            return
        path = Path(directory)
        self.state.background_frames_dir = path
        self.state.background = None
        self.statusBar().showMessage(f"Background frames set to {path.name}")
        self._refresh_plot()
        self._fit_view()
        self._update_control_states()
        self._refresh_background_warning()
        self._update_background_summary()

    def _on_template_context_menu(self, pos) -> None:
        item = self.cell_template_list.itemAt(pos)
        selected_items = self.cell_template_list.selectedItems()
        selected_items = [
            it for it in selected_items if not self.asset_browser.is_add_template_item(it)
        ]
        if item is not None and self.asset_browser.is_add_template_item(item):
            item = None
        menu = QMenu(self.cell_template_list)

        add_action = menu.addAction("Add to Scenario")
        self._apply_action_icon(add_action, "add")
        edit_mask_action = menu.addAction("Edit Mask…")
        self._apply_action_icon(edit_mask_action, "edit")

        if item is None and not selected_items:
            add_action.setEnabled(False)

        target_for_edit = item or (selected_items[0] if selected_items else None)
        if target_for_edit is None:
            edit_mask_action.setEnabled(False)

        action = menu.exec(self.cell_template_list.mapToGlobal(pos))
        if action == add_action:
            if item is not None and not selected_items:
                self._on_cell_template_double_clicked(item)
            else:
                self._add_selected_cells()
        elif action == edit_mask_action and target_for_edit is not None:
            template_path = self.asset_controller.get_path_from_item(target_for_edit)
            if template_path is None:
                return
            mask_path = find_mask_for_template(template_path)
            if mask_path is None:
                QMessageBox.warning(
                    self,
                    "Edit Mask",
                    f"No mask found matching {template_path.name}",
                )
                return
            self._launch_mask_editor(template_path, mask_path)

    def _set_background(self, path: Path) -> None:
        self.state.background = path
        self.state.background_frames_dir = None
        self.statusBar().showMessage(f"Background set to {path.name}")
        self._refresh_plot()
        self._fit_view()
        self._refresh_background_warning()
        self._update_background_summary()
        self._update_control_states()  # Update UI state

    def _on_cell_line_changed(self, line_name: str) -> None:
        if not line_name or line_name == self.asset_browser.placeholder_text:
            self.current_line = None
            self.cell_template_list.clear()
            return
        self._populate_templates_for_line(line_name)

    def _populate_templates_for_line(self, line_name: Optional[str]) -> None:
        """Populate template list for a cell line and append the add-tile."""
        self.cell_template_list.clear()
        if not line_name:
            self.current_line = None
            return
        self.current_line = line_name
        self.asset_controller.populate_templates_for_line(line_name, self.cell_template_list)
        self.asset_browser.ensure_add_template_tile()

    def _icon_with_size(self, path: Path) -> Tuple[Optional[QIcon], Tuple[int, int]]:
        # Wrapper for backward compatibility
        return self.asset_controller.get_icon_with_size(path)

    def _on_cell_template_double_clicked(self, item: QListWidgetItem) -> None:
        """Add cell to scenario when double-clicked."""
        template = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(template, Path):
            return
        added_ids = self.cell_controller.add_cells_from_templates(
            [template],
            refresh_callback=lambda cell_id: self._refresh_scenario_list(select_id=cell_id),
        )
        if not added_ids:
            return
        self.statusBar().showMessage(f"Added {', '.join(added_ids)}")
        self._refresh_plot()  # Will show canvas only if background exists
        self._update_control_states()  # Update UI state

    def _add_selected_cells(self) -> None:
        items = self.cell_template_list.selectedItems()
        if not items:
            return
        template_paths = [
            self.asset_controller.get_path_from_item(item)
            for item in items
        ]
        template_paths = [p for p in template_paths if p is not None]

        added_ids = self.cell_controller.add_cells_from_templates(
            template_paths,
            refresh_callback=lambda cell_id: self._refresh_scenario_list(select_id=cell_id),
        )
        if added_ids:
            self.statusBar().showMessage(f"Added {', '.join(added_ids)}")
            self._refresh_plot()

    def _on_add_template_requested(self) -> None:
        """Launch file picker and mask editor for a new cell template."""
        if not getattr(self, "current_line", None):
            QMessageBox.information(self, "Add Template", "Select a cell line first.")
            return
        line_name = self.current_line
        asset_base = asset_root() / "cell_lines" / line_name
        cells_dir = asset_base / "cells"
        masks_dir = asset_base / "masks"
        cells_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        filter_parts = ["PNG Images (*.png)"]
        if IMAGE_GLOBS:
            filter_parts.append(f"All supported ({' '.join(IMAGE_GLOBS)})")
        filter_parts.append("All Files (*)")
        filter_str = ";;".join(filter_parts)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select cell template image",
            str(cells_dir),
            filter_str,
        )
        if not file_path:
            return

        source_path = Path(file_path)
        try:
            image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        except Exception as exc:
            logger.exception("Failed to read template image: %s", exc)
            QMessageBox.critical(self, "Add Template", f"Could not load {source_path.name}")
            return

        if image is None or image.size == 0:
            QMessageBox.warning(self, "Add Template", f"Could not load {source_path.name}")
            return

        next_idx = self._next_template_index(cells_dir)
        image_name = f"cell_{next_idx}.png"
        mask_name = f"cell_{next_idx}_mask.png"

        temp_dir = Path(tempfile.mkdtemp(prefix="cell_template_import_"))
        temp_image_path = temp_dir / image_name
        temp_mask_path = temp_dir / mask_name
        try:
            if not cv2.imwrite(str(temp_image_path), image):
                raise IOError(f"Failed to write temporary image to {temp_image_path}")
            blank_mask = np.zeros_like(image, dtype=np.uint8)
            if not cv2.imwrite(str(temp_mask_path), blank_mask):
                raise IOError(f"Failed to write temporary mask to {temp_mask_path}")

            dialog = MaskEditorDialog(temp_image_path, temp_mask_path, self)
            if hasattr(dialog, "brush_button"):
                dialog.brush_button.setChecked(True)
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted:
                return

            dest_image_path = cells_dir / image_name
            dest_mask_path = masks_dir / mask_name
            dest_image_path.parent.mkdir(parents=True, exist_ok=True)
            dest_mask_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_image_path, dest_image_path)
            shutil.copy2(temp_mask_path, dest_mask_path)
        except Exception as exc:
            logger.exception("Failed to add template", exc_info=exc)
            QMessageBox.critical(self, "Add Template", str(exc))
            return
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self._populate_templates_for_line(self.current_line)
        self.statusBar().showMessage(f"Added {image_name}")

    def _next_template_index(self, cells_dir: Path) -> int:
        """Return the next sequential template index within a cell line."""
        max_idx = -1
        if cells_dir.exists():
            for path in cells_dir.iterdir():
                if not path.is_file():
                    continue
                stem = path.stem
                if not stem.startswith("cell_"):
                    continue
                try:
                    idx = int(stem.split("cell_", 1)[1])
                except (ValueError, IndexError):
                    continue
                max_idx = max(max_idx, idx)
        return max_idx + 1

    def _remove_selected_cells(self) -> None:
        selected = self.scenario_cell_list.selectedItems()
        if not selected:
            return
        rows = sorted({self.scenario_cell_list.row(item) for item in selected}, reverse=True)
        removed = self.cell_controller.remove_cells_by_indices(rows)
        self.selection_state.clear()
        self._refresh_scenario_list()
        if removed:
            self.statusBar().showMessage(f"Removed {', '.join(removed)}")
        self._refresh_plot()
        self._update_control_states()  # Update UI state

    def _refresh_scenario_list(self, select_id: Optional[str] = None) -> None:
        self.scenario_cell_list.clear()
        target_row: Optional[int] = None
        for row, cell in enumerate(self.state.cells):
            item = QListWidgetItem(cell.id)
            self.scenario_cell_list.addItem(item)
            self.cell_inspector.decorate_scenario_item(item)
            if select_id and cell.id == select_id:
                target_row = row
        if select_id and target_row is not None:
            self.selection_state.select(target_row)
        elif self.state.cells and not self.selection_state.selected:
            self.selection_state.select(0)
        elif self.state.cells:
            self._apply_selection_to_scenario_list(self.selection_state.snapshot())
        else:
            self._apply_selection_to_scenario_list(SelectionSnapshot(None, frozenset(), None))
        if not self.state.cells:
            self.selection_state.clear()
        if hasattr(self.cell_inspector, "set_preview_visible"):
            self.cell_inspector.set_preview_visible(bool(self.state.cells))

    def _on_scenario_cell_selected(self) -> None:
        # Tier 1: Auto-stop preview when selecting a different cell
        self._auto_stop_preview_for_edit("cell selection")
        if not self.scenario_cell_list:
            self.selection_state.clear()
            return
        rows = sorted(
            {
                self.scenario_cell_list.row(item)
                for item in self.scenario_cell_list.selectedItems()
            }
        )
        rows = [row for row in rows if row >= 0]
        if not rows:
            self.selection_state.clear()
            return
        primary = self.scenario_cell_list.currentRow()
        if primary not in rows:
            primary = rows[-1]
        self.selection_state.replace(rows, primary)

    def _on_scenario_context_menu(self, pos) -> None:
        item = self.scenario_cell_list.itemAt(pos)
        menu = QMenu(self.scenario_cell_list)
        edit_mask_action = menu.addAction("Edit Mask…")
        self._apply_action_icon(edit_mask_action, "edit")
        rename_action = menu.addAction("Rename…")
        self._apply_action_icon(rename_action, "edit")
        remove_action = menu.addAction("Remove Cell")
        self._apply_action_icon(remove_action, "delete")
        if item is None:
            edit_mask_action.setEnabled(False)
            rename_action.setEnabled(False)
            remove_action.setEnabled(False)
        action = menu.exec(self.scenario_cell_list.mapToGlobal(pos))
        if action == edit_mask_action and item is not None:
            self.scenario_cell_list.setCurrentItem(item)
            self._open_mask_editor_for_selection()
        elif action == rename_action and item is not None:
            row = self.scenario_cell_list.row(item)
            self._rename_scenario_cell(row)
        elif action == remove_action and item is not None:
            self.scenario_cell_list.setCurrentItem(item)
            self._remove_selected_cells()

    def _rename_scenario_cell(self, row: int) -> None:
        if row < 0 or row >= len(self.state.cells):
            return
        cell = self.state.cells[row]
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Cell",
            "Cell name:",
            text=cell.id,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, "Rename Cell", "Name cannot be empty.")
            return
        if any(other.id == new_name and other is not cell for other in self.state.cells):
            QMessageBox.warning(self, "Rename Cell", "Another cell already uses that name.")
            return
        if new_name == cell.id:
            return
        cell.id = new_name
        self._refresh_scenario_list(select_id=new_name)
        self.statusBar().showMessage(f"Renamed cell to {new_name}")

    # Trajectory editing --------------------------------------------------

    def _current_cell_index(self) -> int:
        if getattr(self, "selection_state", None) and self.selection_state.primary is not None:
            return self.selection_state.primary
        if self.scenario_cell_list is None:
            return -1
        return self.scenario_cell_list.currentRow()

    def _current_cell(self) -> Optional[ScenarioCell]:
        idx = self._current_cell_index()
        if idx < 0 or idx >= len(self.state.cells):
            return None
        return self.state.cells[idx]

    def _set_spin_value(self, spin: QDoubleSpinBox, value: float) -> None:
        was_blocked = spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(was_blocked)

    def _update_coordinate_fields_from_cell(self, cell: ScenarioCell) -> None:
        # Import trajectory metadata for generic UI hints
        from ..core.trajectory import get_trajectory_metadata
        
        # Get metadata for the current trajectory type
        meta = get_trajectory_metadata(cell.trajectory_type)
        
        # Update label text based on trajectory metadata
        self.start_label.setText(meta.start_label)
        
        self._set_spin_value(self.start_x_spin, cell.start[0])
        self._set_spin_value(self.start_y_spin, cell.start[1])
        
        # Hide/show end coordinates based on trajectory metadata
        if hasattr(self.cell_inspector, "_set_row_visibility"):
            self.cell_inspector._set_row_visibility("end", meta.show_end)
        if meta.show_end:
            self.end_label.setText(meta.end_label)
            self._set_spin_value(self.end_x_spin, cell.end[0])
            self._set_spin_value(self.end_y_spin, cell.end[1])

        # Control points visibility based on metadata
        control_points = cell.control_points
        show_ctrl1 = meta.show_controls and meta.num_control_points >= 1 and len(control_points) >= 1
        show_ctrl2 = meta.show_controls and meta.num_control_points >= 2 and len(control_points) >= 2
        
        if hasattr(self.cell_inspector, "_set_row_visibility"):
            self.cell_inspector._set_row_visibility("control1", show_ctrl1)
        if show_ctrl1:
            self._set_spin_value(self.control1_x_spin, control_points[0][0])
            self._set_spin_value(self.control1_y_spin, control_points[0][1])
        if hasattr(self.cell_inspector, "_set_row_visibility"):
            self.cell_inspector._set_row_visibility("control2", show_ctrl2)
        if show_ctrl2:
            self._set_spin_value(self.control2_x_spin, control_points[1][0])
            self._set_spin_value(self.control2_y_spin, control_points[1][1])
        
        # Create dynamic parameter widgets if trajectory supports custom params
        if hasattr(self.cell_inspector, "_create_dynamic_params"):
            current_params = getattr(cell, 'params', {}) or {}
            cell_data = {
                'start': cell.start,
                'end': cell.end,
                'params': current_params,
                'magnification': self.state.magnification,
                'pixel_size_um': self.state.pixel_size_um,
            }
            self.cell_inspector._create_dynamic_params(cell.trajectory_type, current_params, cell_data)

    def _load_selected_cell_details(self, refresh_plot: bool = True) -> None:
        cell = self._current_cell()
        if cell is None:
            self.cell_id_label.setText("Cell: -")
            self._active_curve = None
            self._update_scenario_preview(None)
            return
        self._update_scenario_preview(cell)
        
        # Calculate radius from mask and update label
        try:
            radius_um = calculate_radius_from_mask(cell.mask)
            self.cell_id_label.setText(f"Cell: {cell.id}    •    radius: {radius_um:.2f} µm")
        except Exception:
            # Fallback if radius calculation fails
            self.cell_id_label.setText(f"Cell: {cell.id}")
        
        if hasattr(self.cell_inspector, "set_trajectory_value"):
            self.cell_inspector.set_trajectory_value(cell.trajectory_type)
        else:
            self.trajectory_combo.setCurrentText(cell.trajectory_type)
        cell.ensure_controls(self.state.resolution)
        self._update_coordinate_fields_from_cell(cell)
        self._update_scenario_preview(cell)
        if refresh_plot:
            self._refresh_plot()
        else:
            self._update_active_curve()

    def _update_scenario_preview(self, cell: Optional[ScenarioCell]) -> None:
        preview_label = getattr(self, "cell_preview_label", None)
        edit_button = getattr(self, "cell_preview_edit_button", None)
        if not preview_label:
            return
        if edit_button:
            edit_button.setEnabled(cell is not None)
        if cell is None:
            preview_label.clear()
            return

        try:
            image = cv2.imread(str(cell.template), cv2.IMREAD_GRAYSCALE)
        except Exception:
            image = None
        if image is None:
            preview_label.clear()
        else:
            h, w = image.shape
            qimage = QImage(image.data, w, h, int(image.strides[0]), QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage.copy())
            scaled = pixmap.scaled(preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            preview_label.setText("")
            preview_label.setPixmap(scaled)

    def _on_position_component_changed(self, attribute: str, axis: int, editor: QDoubleSpinBox) -> None:
        # Tier 1: Auto-stop preview when editing trajectory parameters
        self._auto_stop_preview_for_edit("position change")
        
        cell = self._current_cell()
        if cell is None:
            return
        value = editor.value()
        coords = list(getattr(cell, attribute))
        coords[axis] = value
        setattr(cell, attribute, (coords[0], coords[1]))
        
        # Check if this trajectory type has shared_params and we're changing 'end' (amplitude/direction)
        from ..core.trajectory import get_trajectory_metadata
        meta = get_trajectory_metadata(cell.trajectory_type)
        
        if meta.shared_params and attribute == "end":
            # Propagate end position to all cells with the same trajectory type
            # Keep the same relative end offset for all cells
            new_end = (coords[0], coords[1])
            count = 0
            for idx, other_cell in enumerate(self.state.cells):
                if other_cell.trajectory_type == cell.trajectory_type and other_cell is not cell:
                    # Calculate the offset from start to end
                    # Keep the same relative offset for all cells
                    dx = new_end[0] - cell.start[0]
                    dy = new_end[1] - cell.start[1]
                    other_cell.end = (other_cell.start[0] + dx, other_cell.start[1] + dy)
                    count += 1
                    # Update trajectory visuals for this cell
                    if idx == self._current_cell_index():
                        self._sync_selected_cell_geometry(other_cell)
                    else:
                        self._sync_inactive_cell_geometry(idx, other_cell)
            if count > 0:
                self.statusBar().showMessage(
                    f"Updated end position for {count + 1} cells"
                )
        
        self._update_coordinate_fields_from_cell(cell)
        self._refresh_plot()
        warning = self._warn_out_of_bounds(cell)
        if warning:
            self.statusBar().showMessage(warning)
        
        if self._thumbnail_manager:
            cell_idx = self._current_cell_index()
            if cell_idx is not None and cell_idx >= 0:
                self._thumbnail_manager.update_cell_position(cell_idx, cell.start)
        else:
            self.statusBar().showMessage(f"{cell.id} {attribute} updated")

    def _on_control_component_changed(self, index: int, axis: int, editor: QDoubleSpinBox) -> None:
        # Tier 1: Auto-stop preview when editing trajectory parameters
        self._auto_stop_preview_for_edit("control point change")
        
        cell = self._current_cell()
        if cell is None:
            return
        if len(cell.control_points) <= index:
            return
        value = editor.value()
        control_points = list(cell.control_points)
        coords = list(control_points[index])
        coords[axis] = value
        control_points[index] = (coords[0], coords[1])
        cell.control_points = control_points
        self._update_coordinate_fields_from_cell(cell)
        self._refresh_plot()
        warning = self._warn_out_of_bounds(cell)
        if warning:
            self.statusBar().showMessage(warning)
        else:
            self.statusBar().showMessage(f"{cell.id} control point updated")

    def _on_trajectory_param_changed(self, param_name: str, value: Any) -> None:
        """Handle changes to trajectory-specific parameters."""
        from ..core.trajectory import get_trajectory_metadata
        
        # Tier 1: Auto-stop preview when editing trajectory parameters
        self._auto_stop_preview_for_edit("trajectory parameter change")
        
        cell = self._current_cell()
        if cell is None:
            return
        
        # Special case: __end_update__ means update the cell's end position
        # This is used by custom widgets that compute end from other values (e.g., polar coords)
        if param_name == "__end_update__":
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                new_end = (float(value[0]), float(value[1]))
                cell.end = new_end
                # Use the existing propagation mechanism
                meta = get_trajectory_metadata(cell.trajectory_type)
                if meta.shared_params:
                    self._propagate_end_position_to_matching_cells(cell, new_end)
                self._update_coordinate_fields_from_cell(cell)
                self._refresh_plot()
                self.statusBar().showMessage("Updated end position")
            return
        
        # Update the parameter on the current cell
        if not hasattr(cell, 'params') or cell.params is None:
            cell.params = {}
        cell.params[param_name] = value
        
        # Check if this trajectory type has shared_params
        meta = get_trajectory_metadata(cell.trajectory_type)
        if meta.shared_params:
            # Propagate to all cells with the same trajectory type
            count = 0
            for idx, other_cell in enumerate(self.state.cells):
                if other_cell.trajectory_type == cell.trajectory_type and other_cell is not cell:
                    if not hasattr(other_cell, 'params') or other_cell.params is None:
                        other_cell.params = {}
                    other_cell.params[param_name] = value
                    count += 1
                    # Update trajectory visuals for this cell
                    if idx == self._current_cell_index():
                        self._sync_selected_cell_geometry(other_cell)
                    else:
                        self._sync_inactive_cell_geometry(idx, other_cell)
            
            if count > 0:
                self.statusBar().showMessage(
                    f"Updated {param_name} to {value} for {count + 1} cells with {cell.trajectory_type} trajectory"
                )
            else:
                self.statusBar().showMessage(f"Updated {param_name} to {value}")
        else:
            self.statusBar().showMessage(f"{cell.id}: {param_name} = {value}")
        
        # Refresh plot to reflect changes
        self._refresh_plot()

    def _warn_out_of_bounds(self, cell: ScenarioCell) -> Optional[str]:
        width, height = self.state.resolution
        for point in [cell.start, cell.end, *cell.control_points]:
            if point[0] < 0 or point[1] < 0 or point[0] > width or point[1] > height:
                return "Warning: trajectory points extend outside the frame."
        return None

    def _launch_mask_editor(self, template_path: Path, mask_path: Path) -> None:
        dialog = MaskEditorDialog(template_path, mask_path, self)
        dialog.mask_saved.connect(self._handle_mask_saved)
        dialog.exec()

    def _open_mask_editor_for_selection(self) -> None:
        cell = self._current_cell()
        if cell is None:
            QMessageBox.information(
                self,
                "Edit Mask",
                "Select a scenario cell first.",
            )
            return

        self._launch_mask_editor(cell.template, cell.mask)

    def _handle_mask_saved(self, mask_path: Path) -> None:
        mask_path = Path(mask_path)
        affected_cells = [cell for cell in self.state.cells if Path(cell.mask) == mask_path]
        if not affected_cells:
            self.statusBar().showMessage(f"Mask saved to {mask_path.name}")
            return
        if self._thumbnail_manager:
            for cell in affected_cells:
                self._thumbnail_manager.clear_cache_for_cell(cell)

        self._load_selected_cell_details(refresh_plot=False)
        self._refresh_plot()

        names_preview = ", ".join(cell.id for cell in affected_cells[:3])
        suffix = "" if len(affected_cells) <= 3 else "…"
        self.statusBar().showMessage(
            f"Updated mask for {names_preview}{suffix}"
        )

    def _on_trajectory_type_changed(self, trajectory_type: str) -> None:
        # Tier 1: Auto-stop preview when editing trajectory parameters
        self._auto_stop_preview_for_edit("trajectory type change")
        
        cell = self._current_cell()
        if cell is None:
            return
        previous_type = cell.trajectory_type
        cell.trajectory_type = trajectory_type
        resolution = self.state.resolution
        if previous_type == "stationary" and trajectory_type != "stationary":
            cell.end = cell.default_end_for_resolution(resolution)
        cell.ensure_controls(resolution, force=True)
        
        # If the new trajectory type has shared_params, copy params from an existing cell
        from ..core.trajectory import get_trajectory_metadata
        meta = get_trajectory_metadata(trajectory_type)
        if meta.shared_params:
            # Find another cell with the same trajectory type to copy params from
            source_cell = None
            for other_cell in self.state.cells:
                if other_cell is not cell and other_cell.trajectory_type == trajectory_type:
                    source_cell = other_cell
                    break
            
            if source_cell is not None:
                # Copy params from the source cell
                if hasattr(source_cell, 'params') and source_cell.params:
                    if not hasattr(cell, 'params'):
                        cell.params = {}
                    cell.params.update(source_cell.params)
                    
                # Also copy the end offset (amplitude/direction) for consistency
                if source_cell.start and source_cell.end and cell.start:
                    dx = source_cell.end[0] - source_cell.start[0]
                    dy = source_cell.end[1] - source_cell.start[1]
                    cell.end = (cell.start[0] + dx, cell.start[1] + dy)
        
        self._load_selected_cell_details()

    # Video Settings -------------------------------------------------------

    def _on_resolution_changed(self, value: float) -> None:
        """Handle resolution changes."""
        self._auto_stop_preview_for_edit("resolution change")
        
        width = int(self.resolution_width_spin.value())
        height = int(self.resolution_height_spin.value())
        self.state.resolution = (width, height)
        self._update_video_info_label()
        self._refresh_plot()
        self.statusBar().showMessage(f"Resolution set to {width}×{height}")

    def _on_fps_changed(self, value: float) -> None:
        """Handle FPS changes."""
        self._auto_stop_preview_for_edit("FPS change")
        
        self.state.fps = value
        self._update_video_info_label()
        self.schedule_timeline.update_from_state(self.state)
        self._update_timeline_status_label()
        self._refresh_background_warning()
        self.statusBar().showMessage(f"Frame rate set to {value:.1f} fps")

    def _on_magnification_changed(self, value: float) -> None:
        """Handle magnification changes."""
        self._auto_stop_preview_for_edit("magnification change")
        
        self.state.magnification = value
        self._update_magnification_warning()
        self._update_video_info_label()
        self._refresh_plot()  # Update background with zoom
        self._fit_view()
        self.statusBar().showMessage(f"Magnification set to {value:.2f}×")

    def _on_pixel_size_changed(self, value: float) -> None:
        """Handle pixel size changes."""
        self._auto_stop_preview_for_edit("pixel size change")
        
        self.state.pixel_size_um = value
        self._update_video_info_label()
        # Force preview regeneration to reflect new scaling
        if hasattr(self, 'preview_controller') and self.preview_controller:
            self.preview_controller.stop()
        self.statusBar().showMessage(f"Pixel size set to {value:.2f} µm/px")

    def _on_edge_feather_enable_toggled(self, enabled: bool) -> None:
        if self._edge_feather_ui_updating:
            return
        self.state.edge_feather.enabled = enabled
        self._update_edge_feather_enable_button_text()
        self._set_feather_controls_enabled(enabled)
        self._update_edge_feather_preview()

    def _on_edge_feather_inside_spin_changed(self, value: float) -> None:
        if self._edge_feather_ui_updating:
            return
        self.state.edge_feather.set_inside_pixels(value)
        self._update_edge_feather_preview()

    def _on_edge_feather_outside_spin_changed(self, value: float) -> None:
        if self._edge_feather_ui_updating:
            return
        self.state.edge_feather.set_outside_pixels(value)
        self._update_edge_feather_preview()

    def _on_edge_feather_inside_slider_changed(self, slider_value: int) -> None:
        if self._edge_feather_ui_updating:
            return
        px_value = self._edge_slider_to_px(slider_value)
        self.state.edge_feather.set_inside_pixels(px_value)
        self._sync_edge_feather_controls_from_state()

    def _on_edge_feather_outside_slider_changed(self, slider_value: int) -> None:
        if self._edge_feather_ui_updating:
            return
        px_value = self._edge_slider_to_px(slider_value)
        self.state.edge_feather.set_outside_pixels(px_value)
        self._sync_edge_feather_controls_from_state()

    def _on_bg_ref_mag_changed(self, value: float) -> None:
        """Handle background reference magnification changes."""
        self._auto_stop_preview_for_edit("background reference magnification change")
        
        self.state.background_ref_mag = value
        self._update_magnification_warning()
        self._refresh_plot()  # Update background with new zoom
        self.statusBar().showMessage(f"Background ref. mag. set to {value:.2f}×")
    
    def _on_noise_enabled_changed(self, enabled: bool) -> None:
        """Handle noise enable/disable toggle."""
        self._auto_stop_preview_for_edit("noise toggle")
        
        self.state.noise_enabled = enabled
        self._update_noise_button_text()
        self._refresh_plot()  # Update background with/without noise
        status = "enabled" if enabled else "disabled"
        self.statusBar().showMessage(f"Quantization noise {status}")
    
    def _on_noise_stddev_changed(self, value: float) -> None:
        """Handle noise level changes."""
        self._auto_stop_preview_for_edit("noise level change")
        
        self.state.noise_stddev = value
        if self.state.noise_enabled:
            self._refresh_plot()  # Update background with new noise level
        self.statusBar().showMessage(f"Noise level set to {value:.1f} σ")

    def _update_noise_controls_enabled(self) -> None:
        enabled = self.state.noise_enabled
        if hasattr(self, "noise_stddev_spin"):
            self.noise_stddev_spin.setEnabled(enabled)

    def _update_noise_button_text(self) -> None:
        if hasattr(self, "noise_enabled_check"):
            self.noise_enabled_check.setText("Disable Noise" if self.state.noise_enabled else "Enable Noise")
    
    def _update_magnification_warning(self) -> None:
        """Show/hide magnification warning based on current values."""
        should_warn = self.state.magnification < self.state.background_ref_mag
        self.mag_warning_label.setVisible(should_warn)

    def _update_video_info_label(self) -> None:
        """Update the informational label showing computed video parameters."""
        width, height = self.state.resolution
        fps = self.state.fps
        
        # Calculate actual duration from schedule (0 if empty)
        duration = sum(i.duration_s + i.delay_after_s for i in self.state.schedule) if self.state.schedule else 0.0
        total_frames = int(fps * duration)
        
        # Calculate effective pixel size in micrometers
        effective_px_size = self.state.pixel_size_um / self.state.magnification
        
        video_text = f"<b>Video:</b> {width}×{height} px, {total_frames} frames"
        effective_text = (
            f"<b>Effective pixel size:</b> {effective_px_size:.3f} µm/px "
            f"({self.state.pixel_size_um:.2f} ÷ {self.state.magnification:.1f})"
        )
        
        if hasattr(self, "video_summary_label"):
            self.video_summary_label.setText(video_text)
        if hasattr(self, "effective_pixel_label"):
            self.effective_pixel_label.setText(effective_text)
        self._update_edge_feather_preview()
        self._update_effects_badge()

    def _update_effects_badge(self) -> None:
        if not hasattr(self, "effects_badge"):
            return
        active = bool(self.state.noise_enabled) or bool(getattr(self.state, "edge_feather", None) and self.state.edge_feather.enabled)
        self.effects_badge.setVisible(active)

    def _effective_pixel_size_um(self) -> Optional[float]:
        """Return current effective pixel size (µm/px) if magnification is valid."""
        if self.state.magnification <= 0:
            return None
        if self.state.pixel_size_um <= 0:
            return None
        return self.state.pixel_size_um / self.state.magnification

    def _edge_feather_slider_min_value(self) -> int:
        return int(round(self.state.edge_feather.min_pixels * EDGE_FEATHER_SLIDER_SCALE))

    def _edge_feather_slider_max_value(self) -> int:
        return int(round(self.state.edge_feather.max_pixels * EDGE_FEATHER_SLIDER_SCALE))

    def _px_to_edge_slider(self, px_value: float) -> int:
        clamped = max(
            self.state.edge_feather.min_pixels,
            min(self.state.edge_feather.max_pixels, px_value),
        )
        return int(round(clamped * EDGE_FEATHER_SLIDER_SCALE))

    def _edge_slider_to_px(self, slider_value: int) -> float:
        return slider_value / EDGE_FEATHER_SLIDER_SCALE

    def _sync_edge_feather_controls_from_state(self) -> None:
        """Refresh all edge feather controls from the session state."""
        if not hasattr(self, "edge_feather_enable_button"):
            return

        cfg = self.state.edge_feather

        self._edge_feather_ui_updating = True
        try:
            self.edge_feather_enable_button.setChecked(cfg.enabled)
            self._update_edge_feather_enable_button_text()

            self.edge_feather_inside_spin.setRange(cfg.min_pixels, cfg.max_pixels)
            self.edge_feather_outside_spin.setRange(cfg.min_pixels, cfg.max_pixels)

            self.edge_feather_inside_spin.blockSignals(True)
            self.edge_feather_inside_spin.setValue(cfg.inside_pixels)
            self.edge_feather_inside_spin.blockSignals(False)

            self.edge_feather_outside_spin.blockSignals(True)
            self.edge_feather_outside_spin.setValue(cfg.outside_pixels)
            self.edge_feather_outside_spin.blockSignals(False)

            self.edge_feather_inside_slider.blockSignals(True)
            self.edge_feather_inside_slider.setValue(self._px_to_edge_slider(cfg.inside_pixels))
            self.edge_feather_inside_slider.blockSignals(False)

            self.edge_feather_outside_slider.blockSignals(True)
            self.edge_feather_outside_slider.setValue(self._px_to_edge_slider(cfg.outside_pixels))
            self.edge_feather_outside_slider.blockSignals(False)
        finally:
            self._edge_feather_ui_updating = False
        self._set_feather_controls_enabled(cfg.enabled)
        self._update_edge_feather_preview()

    def _format_edge_feather_um_label(self, um_value: Optional[float]) -> str:
        if um_value is None:
            return "≈ N/A"
        return f"≈ {um_value:.2f} µm"

    def _update_edge_feather_preview(self) -> None:
        """Update µm hints and warning badge for edge feathering."""
        if not hasattr(self, "edge_feather_inside_slider"):
            return
        cfg = self.state.edge_feather
        effective_um = self._effective_pixel_size_um()

        inside_um = cfg.inside_microns(effective_um)
        outside_um = cfg.outside_microns(effective_um)

        if hasattr(self, "edge_feather_inside_um_label"):
            self.edge_feather_inside_um_label.setText(self._format_edge_feather_um_label(inside_um))
        if hasattr(self, "edge_feather_outside_um_label"):
            self.edge_feather_outside_um_label.setText(self._format_edge_feather_um_label(outside_um))

        self.edge_feather_inside_slider.blockSignals(True)
        self.edge_feather_inside_slider.setValue(self._px_to_edge_slider(cfg.inside_pixels))
        self.edge_feather_inside_slider.blockSignals(False)

        self.edge_feather_outside_slider.blockSignals(True)
        self.edge_feather_outside_slider.setValue(self._px_to_edge_slider(cfg.outside_pixels))
        self.edge_feather_outside_slider.blockSignals(False)

        warn = effective_um is None
        self.edge_feather_warning_label.setVisible(warn)
        self._refresh_thumbnails_after_feather_change()

    def _update_edge_feather_enable_button_text(self) -> None:
        if not hasattr(self, "edge_feather_enable_button"):
            return
        self.edge_feather_enable_button.setText(
            "Disable Feather" if self.state.edge_feather.enabled else "Enable Feather"
        )

    def _set_feather_controls_enabled(self, enabled: bool) -> None:
        """Keep edge-feather controls interactive even when disabled in the preview."""
        hint = ""
        if not enabled:
            hint = "Feathering disabled — enable above to apply effect (values remain editable)."

        for widget_name in [
            "edge_feather_inside_spin",
            "edge_feather_inside_slider",
            "edge_feather_outside_spin",
            "edge_feather_outside_slider",
            "edge_feather_inside_um_label",
            "edge_feather_outside_um_label",
        ]:
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(True)
                widget.setToolTip(hint)

    def _edge_feather_runtime_params(self) -> Optional[FeatherParameters]:
        """Return render-ready feather parameters if configuration is valid."""
        cfg = self.state.edge_feather
        if not cfg.enabled:
            return None
        return cfg.to_runtime_params()

    # Schedule -------------------------------------------------------------

    def _refresh_schedule_table(self) -> None:
        self._schedule_table_refreshing = True
        self.schedule_table.blockSignals(True)
        try:
            self.schedule_table.setRowCount(len(self.state.schedule))
            for row, interval in enumerate(self.state.schedule):
                header_item = QTableWidgetItem(str(row + 1))
                header_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.schedule_table.setVerticalHeaderItem(row, header_item)
                self._set_schedule_item(row, 0, interval.frequency_khz)
                self._set_schedule_item(row, 1, interval.duration_s)
                self._set_schedule_item(row, 2, interval.delay_after_s)
                self._set_schedule_item(row, 3, interval.angular_velocity_rad_s)
                self._add_schedule_row_actions(row)
        finally:
            self.schedule_table.blockSignals(False)
            self._schedule_table_refreshing = False
        self.schedule_timeline.update_from_state(self.state)
        self._update_timeline_status_label()
        self._refresh_background_warning()


    def _set_schedule_item(self, row: int, column: int, value: float) -> None:
        item = self.schedule_table.item(row, column)
        is_new = item is None
        if is_new:
            item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.EditRole, float(value))
        item.setData(Qt.ItemDataRole.DisplayRole, float(value))
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if is_new:
            self.schedule_table.setItem(row, column, item)


    def _add_schedule_row_actions(self, row: int) -> None:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        def get_current_row():
            """Find the current row by searching for this container widget."""
            for r in range(self.schedule_table.rowCount()):
                if self.schedule_table.cellWidget(r, 4) is container:
                    return r
            return -1

        actions_button = QToolButton()
        actions_button.setText("\u22EE")
        actions_button.setToolTip("Actions")
        actions_button.setAutoRaise(True)
        actions_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        menu = QMenu(actions_button)
        menu.addAction("Insert Below", lambda: self._insert_schedule_interval(get_current_row()))
        menu.addAction("Duplicate", lambda: self._duplicate_schedule_interval(get_current_row()))
        menu.addAction("Delete", lambda: self._delete_schedule_interval(get_current_row()))
        actions_button.setMenu(menu)

        layout.addWidget(actions_button)
        layout.addStretch(1)
        self.schedule_table.setCellWidget(row, 4, container)

    def _clear_handles(self):
        """Clear all handles, markers, and guide lines."""
        # Clear via HandleManager (if available)
        if self._handle_manager:
            self._handle_manager.clear_all()
        
        # Also clear the backward-compatible dictionaries
        for handle_map in self._handles.values():
            for roi in handle_map.values():
                self.plot_item.removeItem(roi)
        self._handles.clear()
        
        for marker_map in self._handle_markers.values():
            for marker in marker_map.values():
                self.plot_item.removeItem(marker)
        self._handle_markers.clear()
        
        for guide_list in self._guide_lines.values():
            for guide_line in guide_list:
                self.plot_item.removeItem(guide_line)
        self._guide_lines.clear()
        
        for markers in self._inactive_markers.values():
            for marker in markers:
                self.plot_item.removeItem(marker)
        self._inactive_markers.clear()
        self._handles_disabled.clear()
        self._inactive_trajectories.clear()

    def _update_inactive_trajectory_states(self, exclude: Optional[pg.PlotCurveItem] = None) -> None:
        for trajectory in self._inactive_trajectories.values():
            if exclude is not None and trajectory is exclude:
                continue
            if getattr(trajectory, "_is_hovered", False):
                trajectory._set_hover_state(False)
                self._release_curve_hover(getattr(trajectory, "cell_index", None))

    def _claim_curve_hover(self, cell_idx: Optional[int], z_value: float) -> bool:
        if cell_idx is None:
            return False
        if self._hover_owner is None or self._hover_owner == cell_idx:
            self._hover_owner = cell_idx
            self._hover_owner_z = z_value
            return True
        if z_value > self._hover_owner_z + 1e-3:
            self._hover_owner = cell_idx
            self._hover_owner_z = z_value
            return True
        return False

    def _release_curve_hover(self, cell_idx: Optional[int]) -> None:
        if cell_idx is None:
            return
        if self._hover_owner == cell_idx:
            self._hover_owner = None
            self._hover_owner_z = float("-inf")

    def _on_canvas_mouse_moved(self, pos) -> None:
        if hasattr(pos, "scenePos"):
            self._last_scene_pos = pos.scenePos()
        elif isinstance(pos, QPointF):
            self._last_scene_pos = pos

    def _on_thumbnail_mouse_press(self, cell_idx: int, event) -> bool:
        if not self._thumbnails_enabled or cell_idx < 0 or cell_idx >= len(self.state.cells):
            return False
        vb = getattr(self.plot_item, "vb", None)
        if vb is None:
            return False
        cell = self.state.cells[cell_idx]
        self._auto_stop_preview_for_edit("start handle drag")
        scene_pos = event.scenePos()
        handle_map = self._handles.get(cell_idx, {})
        for key, handle in handle_map.items():
            if handle is None:
                continue
            if self._is_point_near_roi(scene_pos, handle):
                if key != "start":
                    return False
                break
        curve_item = self._trajectory_items.get(cell_idx)
        if curve_item is not None and hasattr(curve_item, "_is_point_near_curve"):
            if curve_item._is_point_near_curve(scene_pos):
                start_handle = handle_map.get("start")
                if start_handle is None or not self._is_point_near_roi(scene_pos, start_handle):
                    return False
        data_pos = vb.mapSceneToView(scene_pos)
        self._thumbnail_drag_state = {
            "cell_idx": cell_idx,
            "scene_press": scene_pos,
            "data_press": data_pos,
            "start_point": tuple(cell.start),
        }
        if self.selection_state.primary != cell_idx:
            self._select_cell_from_canvas(cell_idx, event.modifiers())
        return True

    def _on_thumbnail_mouse_move(self, cell_idx: int, event) -> bool:
        return self._apply_thumbnail_drag(cell_idx, event.scenePos())

    def _on_thumbnail_mouse_release(self, cell_idx: int, event) -> bool:
        state = self._thumbnail_drag_state
        if state is None or state.get("cell_idx") != cell_idx:
            return False
        self._apply_thumbnail_drag(cell_idx, event.scenePos())
        self._thumbnail_drag_state = None
        if self._current_cell_index() == cell_idx:
            self._handle_drag_finished()
        return True

    def _apply_thumbnail_drag(self, cell_idx: int, scene_pos: QPointF) -> bool:
        state = self._thumbnail_drag_state
        if state is None or state.get("cell_idx") != cell_idx:
            return False
        vb = getattr(self.plot_item, "vb", None)
        if vb is None:
            return False
        current = vb.mapSceneToView(scene_pos)
        press_data = state["data_press"]
        delta_x = current.x() - press_data.x()
        delta_y = current.y() - press_data.y()
        start_x, start_y = state["start_point"]
        self._set_cell_point(cell_idx, "start", (start_x + delta_x, start_y + delta_y))
        return True

    def _is_mouse_near_selected_handles(self) -> bool:
        if self._last_scene_pos is None:
            return False

        cell_idx = self._current_cell_index()
        if cell_idx < 0:
            return False

        if cell_idx in self._handles:
            for handle in self._handles[cell_idx].values():
                if handle is not None and self._is_point_near_roi(self._last_scene_pos, handle):
                    return True

        if cell_idx in self._handle_markers:
            for marker in self._handle_markers[cell_idx].values():
                if marker is not None and self._is_point_near_marker(self._last_scene_pos, marker):
                    return True

        return False

    def _is_point_near_roi(self, scene_pos: QPointF, roi: pg.ROI) -> bool:
        try:
            rect = roi.sceneBoundingRect().adjusted(-5, -5, 5, 5)
        except Exception:
            return False
        return rect.contains(scene_pos)

    def _is_point_near_marker(self, scene_pos: QPointF, marker: pg.ScatterPlotItem) -> bool:
        try:
            return bool(marker.pointsAt(scene_pos))
        except Exception:
            return False

    def _begin_curve_drag(self, cell_idx: Optional[int] = None) -> None:
        self._curve_drag_stack += 1
        if self._curve_drag_stack == 1:
            vb = getattr(self.plot_item, "vb", None)
            if vb is not None:
                vb.setMouseEnabled(x=False, y=False)
        if cell_idx is not None:
            self._set_handle_interaction(cell_idx, enable=False)

    def _end_curve_drag(self, cell_idx: Optional[int] = None) -> None:
        if self._curve_drag_stack <= 0:
            self._curve_drag_stack = 0
            return
        self._curve_drag_stack -= 1
        if self._curve_drag_stack == 0:
            vb = getattr(self.plot_item, "vb", None)
            if vb is not None:
                vb.setMouseEnabled(x=True, y=True)
        if cell_idx is not None:
            self._set_handle_interaction(cell_idx, enable=True)

    def _set_handle_interaction(self, cell_idx: int, enable: bool) -> None:
        handle_map = self._handles.get(cell_idx, {})
        for handle in handle_map.values():
            if handle is None:
                continue
            handle.setAcceptedMouseButtons(
                Qt.MouseButton.LeftButton if enable else Qt.MouseButton.NoButton
            )
        if enable:
            self._handles_disabled.discard(cell_idx)
        else:
            self._handles_disabled.add(cell_idx)

    def _remove_inactive_markers(self, idx: int) -> None:
        markers = self._inactive_markers.pop(idx, [])
        for marker in markers:
            self.plot_item.removeItem(marker)

    def _create_inactive_markers(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.state.cells):
            return
        cell = self.state.cells[idx]
        self._remove_inactive_markers(idx)
        markers: List[pg.ScatterPlotItem] = []
        start_marker = pg.ScatterPlotItem(
            [cell.start[0]],
            [cell.start[1]],
            symbol="o",
            brush=None,
            size=8,
            pen=pg.mkPen(color=(100, 200, 100, 120), width=1.5),
        )
        self.plot_item.addItem(start_marker)
        markers.append(start_marker)
        if cell.trajectory_type != "stationary":
            end_marker = pg.ScatterPlotItem(
                [cell.end[0]],
                [cell.end[1]],
                symbol="s",
                brush=None,
                size=8,
                pen=pg.mkPen(color=(200, 100, 100, 120), width=1.5),
            )
            self.plot_item.addItem(end_marker)
            markers.append(end_marker)
        self._inactive_markers[idx] = markers

    def _detach_handles_for_cell(self, idx: int, add_inactive: bool = False) -> None:
        for handle in self._handles.pop(idx, {}).values():
            self.plot_item.removeItem(handle)
        for marker in self._handle_markers.pop(idx, {}).values():
            self.plot_item.removeItem(marker)
        for guide in self._guide_lines.pop(idx, []):
            self.plot_item.removeItem(guide)
        if add_inactive:
            self._create_inactive_markers(idx)

    def _attach_handles_for_cell(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.state.cells):
            return
        cell = self.state.cells[idx]
        self._remove_inactive_markers(idx)
        self._handles[idx] = {}
        self._handle_markers[idx] = {}
        self._guide_lines[idx] = []

        if len(cell.control_points) >= 1:
            guide1 = self.plot_item.plot(
                [cell.start[0], cell.control_points[0][0]],
                [cell.start[1], cell.control_points[0][1]],
                pen=pg.mkPen(color=(100, 255, 255, 80), width=2.5, style=Qt.PenStyle.DashLine, dash=[3, 3]),
            )
            self._guide_lines[idx].append(guide1)
            if len(cell.control_points) == 1:
                guide2 = self.plot_item.plot(
                    [cell.control_points[0][0], cell.end[0]],
                    [cell.control_points[0][1], cell.end[1]],
                    pen=pg.mkPen(color=(100, 255, 255, 80), width=2.5, style=Qt.PenStyle.DashLine, dash=[3, 3]),
                )
                self._guide_lines[idx].append(guide2)

        if len(cell.control_points) >= 2:
            guide2 = self.plot_item.plot(
                [cell.control_points[1][0], cell.end[0]],
                [cell.control_points[1][1], cell.end[1]],
                pen=pg.mkPen(color=(100, 255, 255, 80), width=2.5, style=Qt.PenStyle.DashLine, dash=[3, 3]),
            )
            self._guide_lines[idx].append(guide2)
            guide3 = self.plot_item.plot(
                [cell.control_points[0][0], cell.control_points[1][0]],
                [cell.control_points[0][1], cell.control_points[1][1]],
                pen=pg.mkPen(color=(100, 255, 255, 80), width=2.5, style=Qt.PenStyle.DashLine, dash=[3, 3]),
            )
            self._guide_lines[idx].append(guide3)

        start_marker = pg.ScatterPlotItem(
            [cell.start[0]],
            [cell.start[1]],
            symbol="o",
            brush=(0, 255, 0, 89),
            size=11,
            pen=pg.mkPen(color=(0, 255, 0, 180), width=2),
        )
        start_marker.setZValue(50)  # High Z for visibility, but below handles
        self.plot_item.addItem(start_marker)
        self._handle_markers[idx]["start"] = start_marker
        start_handle = self._create_handle(
            cell.start,
            (0, 255, 0, 0),
            lambda pos, c=cell: self._update_cell_point(c, "start", pos),
            size=11,  # Match marker size to avoid blocking trajectory clicks
        )
        self._handles[idx]["start"] = start_handle

        if cell.trajectory_type != "stationary":
            end_marker = pg.ScatterPlotItem(
                [cell.end[0]],
                [cell.end[1]],
                symbol="s",
                brush=(255, 0, 0, 89),
                size=11,
                pen=pg.mkPen(color=(255, 0, 0, 180), width=2),
            )
            end_marker.setZValue(50)  # High Z for visibility, but below handles
            self.plot_item.addItem(end_marker)
            self._handle_markers[idx]["end"] = end_marker
            end_handle = self._create_handle(
                cell.end,
                (255, 0, 0, 0),
                lambda pos, c=cell: self._update_cell_point(c, "end", pos),
                size=11,  # Match marker size to avoid blocking trajectory clicks
            )
            self._handles[idx]["end"] = end_handle

        for cp_idx, cp in enumerate(cell.control_points):
            cp_marker = pg.ScatterPlotItem(
                [cp[0]],
                [cp[1]],
                symbol="d",
                brush=(0, 255, 255, 89),
                size=10,
                pen=pg.mkPen(color=(0, 255, 255, 150), width=2),
            )
            cp_marker.setZValue(50)  # High Z for visibility, but below handles
            self.plot_item.addItem(cp_marker)
            self._handle_markers[idx][f"control{cp_idx}"] = cp_marker
            cp_handle = self._create_handle(
                cp,
                (0, 255, 255, 0),
                lambda pos, c=cell, i=cp_idx: self._update_control_point(c, i, pos),
                size=10,  # Match marker size to avoid blocking trajectory clicks
            )
            self._handles[idx][f"control{cp_idx}"] = cp_handle

        self._active_curve = self._trajectory_items.get(idx)
        if idx in self._handles_disabled:
            self._set_handle_interaction(idx, enable=False)
        self._sync_handle_hit_areas()

    def _create_handle(self, pos, color, callback, size=12):
        """Create a draggable point handle at the given position.
        
        Args:
            pos: (x, y) position tuple
            color: RGB tuple or color name
            callback: Function to call when handle is moved
            size: Size of the handle in pixels (default: 12)
        """
        # Create a non-resizable ROI centered at the position
        # ROI position is top-left corner, so offset by half size to center it
        roi = pg.ROI([pos[0] - size/2, pos[1] - size/2], [size, size], 
                     pen=pg.mkPen((0, 0, 0, 0)),  # Fully transparent - visual feedback from markers
                     resizable=False, 
                     movable=True,
                     removable=False)
        roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        roi.setAcceptHoverEvents(True)  # Accept hover events to block curve interaction
        roi.setZValue(100)  # Highest Z-value to receive handle clicks before everything else
        
        # Override mouse events to consume them and prevent propagation to curve below
        original_mouse_press = roi.mousePressEvent
        original_mouse_drag = roi.mouseDragEvent
        
        def custom_mouse_press(event):
            event.accept()  # Consume the event to prevent propagation
            original_mouse_press(event)
        
        def custom_mouse_drag(event):
            event.accept()  # Consume drag events
            original_mouse_drag(event)
        
        roi.mousePressEvent = custom_mouse_press
        roi.mouseDragEvent = custom_mouse_drag
        
        # Store sizing metadata so we can keep the ROI footprint constant in screen space
        roi._handle_size = size  # Backward compatibility
        roi._handle_pixel_size = size
        roi._handle_data_size = (float(size), float(size))
        
        # Connect to sigRegionChanged for real-time updates during drag
        # ROI.pos() gives top-left corner, add half size to get center
        def current_center(_roi) -> Tuple[float, float]:
            top_left = roi.pos()
            data_w, data_h = getattr(roi, "_handle_data_size", (size, size))
            center_x = top_left.x() + data_w / 2
            center_y = top_left.y() + data_h / 2
            return center_x, center_y

        def on_move(_roi=None):
            callback(current_center(roi))

        def on_finish(_roi=None):
            callback(current_center(roi))
            self._handle_drag_finished()

        roi.sigRegionChanged.connect(on_move)
        roi.sigRegionChangeFinished.connect(on_finish)
        self.plot_item.addItem(roi)
        return roi

    def _ensure_handle_viewbox_sync(self) -> None:
        """Connect to ViewBox signals so handle hit areas update with zoom/pan."""
        if self._handle_viewbox_connected:
            return
        vb = getattr(self, "plot_item", None)
        if vb is None:
            return
        view_box = getattr(vb, "vb", None)
        if view_box is None:
            return
        if hasattr(view_box, "sigRangeChanged"):
            view_box.sigRangeChanged.connect(self._on_viewbox_transform_changed)
        if hasattr(view_box, "sigResized"):
            view_box.sigResized.connect(self._on_viewbox_transform_changed)
        self._handle_viewbox_connected = True

    def _on_viewbox_transform_changed(self, *args) -> None:
        """Keep handle hit boxes aligned with the current zoom level."""
        self._sync_handle_hit_areas()

    def _current_view_pixel_scale(self) -> Optional[Tuple[float, float]]:
        """Return data-units-per-pixel for the current ViewBox, or None if unavailable."""
        plot_item = getattr(self, "plot_item", None)
        if plot_item is None:
            return None
        view_box = getattr(plot_item, "vb", None)
        if view_box is None:
            return None
        view_width = float(max(view_box.width(), 1.0))
        view_height = float(max(view_box.height(), 1.0))
        try:
            (x_min, x_max), (y_min, y_max) = view_box.viewRange()
        except Exception:
            return None
        span_x = abs(float(x_max) - float(x_min))
        span_y = abs(float(y_max) - float(y_min))
        if view_width <= 0 or view_height <= 0:
            return None
        scale_x = span_x / view_width
        scale_y = span_y / view_height
        return scale_x, scale_y

    def _sync_handle_hit_areas(self) -> None:
        """Resize all handle ROIs so their screen-space footprint stays constant."""
        if not self._handles:
            return
        scale = self._current_view_pixel_scale()
        if not scale:
            return
        for handle_map in self._handles.values():
            for handle in handle_map.values():
                self._update_handle_hit_area(handle, scale)

    def _update_handle_hit_area(
        self,
        handle: Optional[pg.ROI],
        scale: Tuple[float, float],
    ) -> None:
        """Resize a specific handle ROI given the current data-units-per-pixel scale."""
        if handle is None:
            return
        pixel_size = getattr(handle, "_handle_pixel_size", getattr(handle, "_handle_size", None))
        if pixel_size is None:
            return
        scale_x, scale_y = scale
        width = float(scale_x * float(pixel_size))
        height = float(scale_y * float(pixel_size))
        if width <= 0:
            width = HANDLE_MIN_DATA_SPAN
        if height <= 0:
            height = HANDLE_MIN_DATA_SPAN
        center = self._get_handle_center(handle)
        if center is None:
            return
        top_left = QPointF(center[0] - width / 2, center[1] - height / 2)
        was_blocked = handle.blockSignals(True)
        try:
            handle.setSize([width, height])
            handle.setPos(top_left)
        finally:
            handle.blockSignals(was_blocked)
        handle._handle_data_size = (width, height)

    def _get_handle_center(self, handle: pg.ROI) -> Optional[Tuple[float, float]]:
        """Return the current center coordinate for a handle ROI."""
        size = getattr(handle, "_handle_data_size", None)
        if size is None:
            size_value = getattr(handle, "_handle_size", None)
            if size_value is None:
                return None
            size = (float(size_value), float(size_value))
            handle._handle_data_size = size
        top_left = handle.pos()
        return top_left.x() + size[0] / 2, top_left.y() + size[1] / 2

    def _update_guide_lines(self, cell_idx: int, cell: ScenarioCell) -> None:
        """Update the guide lines for a cell based on current positions."""
        if cell_idx not in self._guide_lines:
            return
        
        guide_lines = self._guide_lines[cell_idx]
        
        # Update guide lines based on number of control points
        if len(cell.control_points) >= 1 and len(guide_lines) >= 1:
            # Update line from start to first control point
            guide_lines[0].setData(
                [cell.start[0], cell.control_points[0][0]],
                [cell.start[1], cell.control_points[0][1]]
            )
            
            # For parabolic (1 control point), update line from control point to end
            if len(cell.control_points) == 1 and len(guide_lines) >= 2:
                guide_lines[1].setData(
                    [cell.control_points[0][0], cell.end[0]],
                    [cell.control_points[0][1], cell.end[1]]
                )
        
        if len(cell.control_points) >= 2 and len(guide_lines) >= 2:
            # Update line from second control point to end
            guide_lines[1].setData(
                [cell.control_points[1][0], cell.end[0]],
                [cell.control_points[1][1], cell.end[1]]
            )
            
            # Update line between control points
            if len(guide_lines) >= 3:
                guide_lines[2].setData(
                    [cell.control_points[0][0], cell.control_points[1][0]],
                    [cell.control_points[0][1], cell.control_points[1][1]]
                )

    def _update_cell_point(self, cell, attribute, pos):
        """Update a cell's point attribute from ROI position."""
        # Handle both QPointF and tuple returns
        if hasattr(pos, 'x') and hasattr(pos, 'y'):
            x, y = pos.x(), pos.y()
        else:
            x, y = pos[0], pos[1]
        setattr(cell, attribute, (x, y))
        self._update_coordinate_fields_from_cell(cell)
        self._update_active_curve()
        
        # Update guide lines
        cell_idx = self._current_cell_index()
        if cell_idx >= 0:
            self._update_guide_lines(cell_idx, cell)
            
            # Update marker position
            if cell_idx in self._handle_markers and attribute in self._handle_markers[cell_idx]:
                self._handle_markers[cell_idx][attribute].setData([x], [y])
            
        if self._thumbnail_manager and attribute == "start":
            self._thumbnail_manager.update_cell_position(cell_idx, cell.start)

        warning = self._warn_out_of_bounds(cell)
        if warning:
            self.statusBar().showMessage(warning)
        
        # Propagate end position to other cells with shared_params trajectory
        if attribute == "end":
            from rot_game.plugins import get_trajectory_metadata
            meta = get_trajectory_metadata(cell.trajectory_type)
            if meta and meta.shared_params:
                self._propagate_end_position_to_matching_cells(cell, (x, y))

    def _propagate_end_position_to_matching_cells(self, source_cell, new_end_pos):
        """Propagate end position change to all cells with same shared_params trajectory type.
        
        The end offset vector (end - start) is kept consistent across all cells,
        so each cell uses the same relative end position from its own start.
        """
        # Calculate the end offset vector from the source cell
        dx = new_end_pos[0] - source_cell.start[0]
        dy = new_end_pos[1] - source_cell.start[1]
        
        for idx, other_cell in enumerate(self.state.cells):
            if other_cell is source_cell:
                continue
            if other_cell.trajectory_type != source_cell.trajectory_type:
                continue
            # Apply the same end offset vector to this cell's start position
            other_cell.end = (other_cell.start[0] + dx, other_cell.start[1] + dy)
            # Update geometry (this also updates trajectory curves and handles)
            if idx == self._current_cell_index():
                self._sync_selected_cell_geometry(other_cell)
            else:
                self._sync_inactive_cell_geometry(idx, other_cell)

    def _set_cell_point(self, cell_idx: int, attribute: str, pos: Tuple[float, float]) -> None:
        """Programmatically update a cell point for both active and inactive cells."""
        if cell_idx < 0 or cell_idx >= len(self.state.cells):
            return
        cell = self.state.cells[cell_idx]
        setattr(cell, attribute, (float(pos[0]), float(pos[1])))
        if cell_idx == self._current_cell_index():
            self._sync_selected_cell_geometry(cell)
        else:
            self._sync_inactive_cell_geometry(cell_idx, cell)
            if self._thumbnail_manager and attribute == "start":
                self._thumbnail_manager.update_cell_position(cell_idx, cell.start)

    def _update_control_point(self, cell, index, pos):
        """Update a cell's control point from ROI position."""
        # Handle both QPointF and tuple returns
        if hasattr(pos, 'x') and hasattr(pos, 'y'):
            x, y = pos.x(), pos.y()
        else:
            x, y = pos[0], pos[1]
        control_points = list(cell.control_points)
        control_points[index] = (x, y)
        cell.control_points = control_points
        self._update_coordinate_fields_from_cell(cell)
        self._update_active_curve()
        
        # Update guide lines
        cell_idx = self._current_cell_index()
        if cell_idx >= 0:
            self._update_guide_lines(cell_idx, cell)
            
            # Update marker position
            marker_key = f'control{index}'
            if cell_idx in self._handle_markers and marker_key in self._handle_markers[cell_idx]:
                self._handle_markers[cell_idx][marker_key].setData([x], [y])
        
        warning = self._warn_out_of_bounds(cell)
        if warning:
            self.statusBar().showMessage(warning)

    def _handle_drag_finished(self) -> None:
        cell = self._current_cell()
        if cell is None:
            return
        self._update_coordinate_fields_from_cell(cell)
        warning = self._warn_out_of_bounds(cell)
        if warning:
            self.statusBar().showMessage(warning)

    def _update_active_curve(self) -> None:
        if self._active_curve is None:
            return
        cell = self._current_cell()
        if cell is None:
            return
        path = self._trajectory_path(cell, samples=200)
        if path is not None and len(path) > 1:
            self._active_curve.setData(path[:, 0], path[:, 1])
        else:
            self._active_curve.setData(
                [cell.start[0], cell.end[0]],
                [cell.start[1], cell.end[1]],
            )

    def _reposition_handle(self, handle: Optional[pg.ROI], position: Tuple[float, float]) -> None:
        if handle is None:
            return
        data_size = getattr(handle, "_handle_data_size", None)
        if data_size is None:
            base_size = float(getattr(handle, "_handle_size", 12))
            data_size = (base_size, base_size)
            handle._handle_data_size = data_size
        top_left = QPointF(position[0] - data_size[0] / 2, position[1] - data_size[1] / 2)
        was_blocked = handle.blockSignals(True)
        handle.setPos(top_left)
        handle.blockSignals(was_blocked)

    def _sync_selected_cell_geometry(self, cell: ScenarioCell) -> None:
        """Refresh curve, handles, markers, and inspector fields after a geometry edit."""
        self._update_coordinate_fields_from_cell(cell)
        self._update_active_curve()

        cell_idx = self._current_cell_index()
        if cell_idx < 0:
            return

        self._update_guide_lines(cell_idx, cell)

        handle_map = self._handles.get(cell_idx, {})
        self._reposition_handle(handle_map.get("start"), cell.start)
        self._reposition_handle(handle_map.get("end"), cell.end)
        for cp_idx, point in enumerate(cell.control_points):
            self._reposition_handle(handle_map.get(f"control{cp_idx}"), point)

        marker_map = self._handle_markers.get(cell_idx, {})
        if "start" in marker_map:
            marker_map["start"].setData([cell.start[0]], [cell.start[1]])
        if "end" in marker_map:
            marker_map["end"].setData([cell.end[0]], [cell.end[1]])
        for cp_idx, point in enumerate(cell.control_points):
            key = f"control{cp_idx}"
            if key in marker_map:
                marker_map[key].setData([point[0]], [point[1]])

        warning = self._warn_out_of_bounds(cell)
        if warning:
            self.statusBar().showMessage(warning)
        if self._thumbnail_manager:
            self._thumbnail_manager.update_cell_position(cell_idx, cell.start)

        if self._thumbnail_manager and cell_idx >= 0:
            self._thumbnail_manager.update_cell_position(cell_idx, cell.start)

    def _sync_inactive_cell_geometry(self, cell_idx: int, cell: ScenarioCell) -> None:
        """Refresh inactive trajectory curve and its handles/markers after geometry edit.
        
        This is called when dragging an inactive trajectory to update its visuals in real-time.
        """
        # Update the inactive trajectory curve
        inactive_traj = self._trajectory_items.get(cell_idx)
        if inactive_traj is not None:
            path = self._trajectory_path(cell, samples=200)
            if path is not None and len(path) > 1:
                inactive_traj.setData(path[:, 0], path[:, 1])
            else:
                inactive_traj.setData(
                    [cell.start[0], cell.end[0]],
                    [cell.start[1], cell.end[1]],
                )
        
        # Update handles and markers for this inactive cell
        self._update_guide_lines(cell_idx, cell)
        
        handle_map = self._handles.get(cell_idx, {})
        self._reposition_handle(handle_map.get("start"), cell.start)
        self._reposition_handle(handle_map.get("end"), cell.end)
        for cp_idx, point in enumerate(cell.control_points):
            self._reposition_handle(handle_map.get(f"control{cp_idx}"), point)

        marker_map = self._handle_markers.get(cell_idx, {})
        if "start" in marker_map:
            marker_map["start"].setData([cell.start[0]], [cell.start[1]])
        if "end" in marker_map:
            marker_map["end"].setData([cell.end[0]], [cell.end[1]])
        for cp_idx, point in enumerate(cell.control_points):
            key = f"control{cp_idx}"
            if key in marker_map:
                marker_map[key].setData([point[0]], [point[1]])

        warning = self._warn_out_of_bounds(cell)
        if warning:
            self.statusBar().showMessage(warning)
        if self._thumbnail_manager:
            self._thumbnail_manager.update_cell_position(cell_idx, cell.start)

    def _rebuild_thumbnails(self) -> None:
        """Rebuild thumbnail sprites to reflect current cell states."""
        if not self._thumbnail_manager:
            return
        selected_idx = self.selection_state.primary if hasattr(self, "selection_state") else None
        self._thumbnail_manager.rebuild(
            self.state.cells,
            selected_idx,
            self.state.magnification,
            self.state.pixel_size_um,
            self._edge_feather_runtime_params(),
            self.state.noise_stddev if self.state.noise_enabled else None,
        )

    def _refresh_thumbnails_after_feather_change(self) -> None:
        """Force thumbnails to refresh when feather configuration changes."""
        if not self._thumbnail_manager:
            return
        self._thumbnail_manager.clear_cache()
        self._rebuild_thumbnails()

    def _thumbnail_index_at_scene_pos(self, scene_pos: QPointF) -> Optional[int]:
        if not self._thumbnail_manager or not self._thumbnails_enabled:
            return None
        return self._thumbnail_manager.hit_test_scene_point(scene_pos)

    def _set_thumbnail_hover_override(self, idx: Optional[int]) -> None:
        if not self._thumbnail_manager:
            return
        if self._thumbnail_hover_override == idx:
            return
        self._thumbnail_hover_override = idx
        self._thumbnail_manager.set_manual_hover_idx(idx)

    def _thumbnail_hover_owner(self, scene_pos: QPointF) -> Optional[int]:
        idx = self._thumbnail_index_at_scene_pos(scene_pos)
        if idx is None:
            return None
        handle_map = self._handles.get(idx, {})
        start_handle = handle_map.get("start")
        for key, handle in handle_map.items():
            if handle is None:
                continue
            if self._is_point_near_roi(scene_pos, handle):
                if key == "start":
                    return idx
                return None
        curve = self._trajectory_items.get(idx)
        if curve is not None and hasattr(curve, "_is_point_near_curve"):
            if curve._is_point_near_curve(scene_pos):
                return None
        return idx

    def _promote_inactive_curve(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.state.cells):
            return
        existing = self._inactive_trajectories.pop(idx, None)
        if existing is not None and getattr(existing, "_defer_promotion", False):
            self._active_curve = existing
            self._trajectory_items[idx] = existing
            return
        if existing is not None:
            self.plot_item.removeItem(existing)

        cell = self.state.cells[idx]
        path = self._trajectory_path(cell, samples=200)
        if path is not None and len(path) > 1:
            x_data, y_data = path[:, 0], path[:, 1]
        else:
            x_data = [cell.start[0], cell.end[0]]
            y_data = [cell.start[1], cell.end[1]]

        curve = TrajectoryItem(idx, self, self.selection_state, x_data, y_data)
        curve.setZValue(30)
        self.plot_item.addItem(curve)
        self._trajectory_items[idx] = curve
        self._active_curve = curve

    def _demote_active_curve(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.state.cells):
            return
        if self._active_curve is not None:
            self.plot_item.removeItem(self._active_curve)
            self._active_curve = None

        cell = self.state.cells[idx]
        path = self._trajectory_path(cell, samples=200)
        if path is not None and len(path) > 1:
            x_data, y_data = path[:, 0], path[:, 1]
        else:
            x_data = [cell.start[0], cell.end[0]]
            y_data = [cell.start[1], cell.end[1]]

        curve = SelectableTrajectory(idx, self, self.selection_state, x_data, y_data)
        curve.setZValue(10)
        self.plot_item.addItem(curve)
        self._inactive_trajectories[idx] = curve
        self._trajectory_items[idx] = curve

    def _recreate_active_curve(self, idx: int) -> None:
        """Rebuild the active trajectory item (used after deferred promotions)."""
        if idx < 0 or idx >= len(self.state.cells):
            return
        if self.selection_state.primary != idx:
            return
        if self._active_curve is not None:
            self.plot_item.removeItem(self._active_curve)
        cell = self.state.cells[idx]
        path = self._trajectory_path(cell, samples=200)
        if path is not None and len(path) > 1:
            x_data, y_data = path[:, 0], path[:, 1]
        else:
            x_data = [cell.start[0], cell.end[0]]
            y_data = [cell.start[1], cell.end[1]]
        curve = TrajectoryItem(idx, self, self.selection_state, x_data, y_data)
        curve.setZValue(30)
        self.plot_item.addItem(curve)
        self._trajectory_items[idx] = curve
        self._active_curve = curve

    def _trajectory_path(self, cell, samples=200):
        """Generate trajectory path for visualization."""
        from ..core.trajectory import _generate_path
        from ..config import TrajectoryConfig
        
        config = TrajectoryConfig(
            type=cell.trajectory_type,
            start=cell.start,
            end=cell.end,
            control_points=cell.control_points if cell.control_points else None,
            params=getattr(cell, 'params', None) or None,
        )
        try:
            path = _generate_path(config, samples)
            return path
        except ValueError:
            # If trajectory generation fails, return None
            return None

    def _on_schedule_item_changed(self, item: QTableWidgetItem) -> None:
        if self._schedule_table_refreshing:
            return
        # Only commit if the edited item is within editable columns (0-3)
        if item.column() > 3:
            return
        self._apply_schedule_changes()

    def _apply_schedule_changes(self) -> None:
        intervals: List[ScheduleInterval] = []
        for row in range(self.schedule_table.rowCount()):
            entries = [self.schedule_table.item(row, col) for col in range(4)]
            if any(item is None for item in entries):
                QMessageBox.warning(self, "Schedule", f"Missing value in row {row + 1}")
                self._refresh_schedule_table()
                return
            try:
                freq = float(entries[0].data(Qt.ItemDataRole.EditRole))
                duration = float(entries[1].data(Qt.ItemDataRole.EditRole))
                delay = float(entries[2].data(Qt.ItemDataRole.EditRole))
                omega = float(entries[3].data(Qt.ItemDataRole.EditRole))
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Schedule", f"Invalid numeric value in row {row + 1}")
                self._refresh_schedule_table()
                return
            intervals.append(ScheduleInterval(freq, duration, omega, delay))
        if not intervals:
            QMessageBox.warning(self, "Schedule", "At least one interval is required.")
            self._refresh_schedule_table()
            return
        self.state.schedule = intervals
        # Auto-compute video duration from schedule
        self.state.video_duration_s = sum(i.duration_s + i.delay_after_s for i in intervals)
        self.statusBar().showMessage("Schedule updated")
        self._update_video_info_label()
        self.schedule_timeline.update_from_state(self.state)
        self._update_timeline_status_label()
        self._refresh_schedule_table()
        self._update_control_states()  # Update Play/Export buttons when schedule changes

    def _add_schedule_interval(self) -> None:
        self._insert_schedule_interval(len(self.state.schedule) - 1)

    def _remove_schedule_interval(self) -> None:
        row = self.schedule_table.currentRow()
        if row < 0:
            row = len(self.state.schedule) - 1
        self._delete_schedule_interval(row)

    def _insert_schedule_interval(self, row: int) -> None:
        insert_at = max(0, row + 1)
        new_interval = ScheduleInterval(60.0, 2.0, 3.0)
        self.state.schedule.insert(insert_at, new_interval)
        # Update video duration from schedule
        self.state.video_duration_s = sum(i.duration_s + i.delay_after_s for i in self.state.schedule)
        self._update_video_info_label()
        self._refresh_schedule_table()
        self._update_control_states()  # Update UI when schedule changes

    def _duplicate_schedule_interval(self, row: int) -> None:
        if row < 0 or row >= len(self.state.schedule):
            return
        current = self.state.schedule[row]
        duplicate = ScheduleInterval(
            frequency_khz=current.frequency_khz,
            duration_s=current.duration_s,
            angular_velocity_rad_s=current.angular_velocity_rad_s,
            delay_after_s=current.delay_after_s,
        )
        self.state.schedule.insert(row + 1, duplicate)
        # Update video duration from schedule
        self.state.video_duration_s = sum(i.duration_s + i.delay_after_s for i in self.state.schedule)
        self._update_video_info_label()
        self._refresh_schedule_table()
        self._update_control_states()  # Update UI when schedule changes

    def _delete_schedule_interval(self, row: int) -> None:
        if row < 0 or row >= len(self.state.schedule):
            return
        if len(self.state.schedule) == 1:
            QMessageBox.information(self, "Schedule", "At least one interval is required.")
            return
        self.state.schedule.pop(row)
        # Update video duration from schedule
        self.state.video_duration_s = sum(i.duration_s + i.delay_after_s for i in self.state.schedule)
        self._update_video_info_label()
        self._refresh_schedule_table()
        self._update_control_states()  # Update UI when schedule changes

    def _on_schedule_context_menu(self, pos) -> None:
        row = self.schedule_table.indexAt(pos).row()
        if row < 0:
            return
        menu = QMenu(self.schedule_table)
        insert_action = menu.addAction("Insert Below")
        duplicate_action = menu.addAction("Duplicate")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.schedule_table.viewport().mapToGlobal(pos))
        if action == insert_action:
            self._insert_schedule_interval(row)
        elif action == duplicate_action:
            self._duplicate_schedule_interval(row)
        elif action == delete_action:
            self._delete_schedule_interval(row)

    def eventFilter(self, obj, event):
        if obj in (self.schedule_table, self.schedule_table.viewport()) and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._delete_schedule_interval(self.schedule_table.currentRow())
                return True
        return super().eventFilter(obj, event)

    # Preview & render ----------------------------------------------------

    def _on_preview_frame_advanced(self, frame: int, time_s: float) -> None:
        """Update timeline playhead during preview playback.
        
        Args:
            frame: Current frame number
            time_s: Current time in seconds
        """
        self.schedule_timeline.set_playhead_position(frame, time_s)
        self._update_timeline_status_label(frame, time_s)

    def _on_timeline_seek(self, frame: int, time_s: float) -> None:
        """Handle timeline seek requests (click-to-jump or scrubbing).
        
        Args:
            frame: Target frame number
            time_s: Target time in seconds
        """
        if self.preview_controller and self.preview_controller.is_active():
            # Already in preview mode: pause and seek
            self.preview_controller.pause()
            self.preview_controller.seek(frame)
            self._update_button_state_paused()
            self.statusBar().showMessage(f"Paused at frame {frame} ({time_s:.2f}s)")
        else:
            # Not in preview mode: auto-enter preview mode paused at clicked frame
            self._start_preview_at_frame(frame)
            self.statusBar().showMessage(f"Preview mode: Frame {frame} ({time_s:.2f}s)")
        self._update_timeline_status_label(frame, time_s)
    
    def _on_canvas_mouse_clicked(self, event) -> None:
        """Stop preview when the canvas is clicked during preview mode."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.preview_controller or not self.preview_controller.is_active():
            return

        self._on_stop_clicked()
        if hasattr(event, "accept"):
            event.accept()

    def _select_cell_from_canvas(
        self,
        cell_index: int,
        modifiers: Qt.KeyboardModifiers | None = None,
    ) -> None:
        """Select the scenario cell linked to the clicked trajectory curve."""
        if not self.state.cells:
            return
        if cell_index < 0 or cell_index >= len(self.state.cells):
            return

        if self.scenario_cell_list is None:
            return

        modifiers = modifiers or Qt.KeyboardModifier.NoModifier
        toggle = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        additive = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if toggle:
            self.selection_state.toggle(cell_index)
        else:
            self.selection_state.select(cell_index, additive=additive)

    def _apply_selection_to_scenario_list(self, snapshot: SelectionSnapshot | None = None) -> None:
        if not self.scenario_cell_list:
            return
        snapshot = snapshot or self.selection_state.snapshot()
        blocked = self.scenario_cell_list.blockSignals(True)
        self.scenario_cell_list.clearSelection()
        for idx in sorted(snapshot.selected):
            if 0 <= idx < self.scenario_cell_list.count():
                item = self.scenario_cell_list.item(idx)
                if item is not None:
                    item.setSelected(True)
        if snapshot.primary is not None and 0 <= snapshot.primary < self.scenario_cell_list.count():
            self.scenario_cell_list.setCurrentRow(snapshot.primary)
        elif snapshot.selected:
            fallback = min(snapshot.selected)
            if 0 <= fallback < self.scenario_cell_list.count():
                self.scenario_cell_list.setCurrentRow(fallback)
        else:
            self.scenario_cell_list.setCurrentRow(-1)
        self.scenario_cell_list.blockSignals(blocked)
        if hasattr(self, "cell_inspector") and hasattr(self.cell_inspector, "_update_item_selection_styles"):
            self.cell_inspector._update_item_selection_styles()

    def _on_selection_changed(self, snapshot: SelectionSnapshot) -> None:
        previous = self._current_primary
        new_primary = snapshot.primary
        if previous != new_primary:
            if previous is not None:
                self._demote_active_curve(previous)
                self._detach_handles_for_cell(previous, add_inactive=True)
            if new_primary is not None:
                self._remove_inactive_markers(new_primary)
                self._attach_handles_for_cell(new_primary)
                self._promote_inactive_curve(new_primary)
            else:
                self._active_curve = None
            self._current_primary = new_primary
        if self._thumbnail_manager:
            self._thumbnail_manager.set_selection(new_primary)
        self._apply_selection_to_scenario_list(snapshot)
        self._load_selected_cell_details(refresh_plot=False)
        self._update_control_states()

    def _on_selection_hover_changed(self, hovered_idx: Optional[int]) -> None:
        if self._thumbnail_manager:
            self._thumbnail_manager.set_hovered(hovered_idx)

    def _on_play_pause_clicked(self) -> None:
        """Handle Play/Pause button clicks."""
        if not self.preview_controller:
            return

        if self.preview_controller.is_active():
            # Currently playing - check if timer is running
            if self.preview_controller._timer.isActive():
                # Pause
                self.preview_controller.pause()
                self._update_button_state_paused()
                self.statusBar().showMessage("Preview paused")
            else:
                # Resume
                self.preview_controller.resume()
                self._update_button_state_playing()
                self.statusBar().showMessage("Preview resumed")
        else:
            # Start new preview
            self._start_preview()
    
    def _on_stop_clicked(self) -> None:
        """Handle Stop button clicks."""
        if not self.preview_controller:
            return
        
        if self.preview_controller.is_active():
            self._preview_finish_message = "Preview stopped"
            self.preview_controller.stop()
    
    def _update_button_state_playing(self) -> None:
        """Update button states for playing preview."""
        self._apply_icon(self.play_pause_button, "pause")
        self.play_pause_button.setText("Pause")
        self.stop_button.setEnabled(True)
        if hasattr(self, "play_pause_action"):
            self.play_pause_action.setText("Pause Preview")
            self._apply_icon(self.play_pause_action, "pause")
        if hasattr(self, "stop_action"):
            self.stop_action.setEnabled(True)
    
    def _update_button_state_paused(self) -> None:
        """Update button states for paused preview."""
        self._apply_icon(self.play_pause_button, "play")
        self.play_pause_button.setText("Resume")
        self.stop_button.setEnabled(True)
        if hasattr(self, "play_pause_action"):
            self.play_pause_action.setText("Resume Preview")
            self._apply_icon(self.play_pause_action, "play")
        if hasattr(self, "stop_action"):
            self.stop_action.setEnabled(True)
    
    def _update_button_state_stopped(self) -> None:
        """Update button states for stopped/idle state."""
        self._apply_icon(self.play_pause_button, "play")
        self.play_pause_button.setText("Play")
        self.stop_button.setEnabled(False)
        if hasattr(self, "play_pause_action"):
            self.play_pause_action.setText("Play Preview")
            self._apply_icon(self.play_pause_action, "play")
        if hasattr(self, "stop_action"):
            self.stop_action.setEnabled(False)

    def _load_icon(self, name: str) -> Optional[QIcon]:
        """Load and cache icons from the GUI asset directory."""
        if name in self._icon_cache:
            return self._icon_cache[name]

        path = self._icon_dir / f"{name}.svg"
        if not path.exists():
            logger.warning("Icon asset not found: %s", path)
            return None

        icon = QIcon(str(path))
        self._icon_cache[name] = icon
        return icon

    def _apply_icon(self, target: Any, name: str) -> None:
        """Apply a named icon to any Qt object that exposes setIcon."""
        icon = self._load_icon(name)
        if not icon:
            return

        set_icon = getattr(target, "setIcon", None)
        if callable(set_icon):
            set_icon(icon)

    def _set_tab_icon(self, index: int, name: str) -> None:
        icon = self._load_icon(name)
        if icon:
            self.sidebar_tabs.setTabIcon(index, icon)

    def _apply_action_icon(self, action: QAction, name: str) -> None:
        icon = self._load_icon(name)
        if icon:
            action.setIcon(icon)
            action.setIconVisibleInMenu(True)

    def _set_editing_enabled(self, enabled: bool) -> None:
        """Enable or disable editing widgets during preview mode.
        
        Implements the hybrid approach:
        - Tier 2 (Disabled): Destructive changes that invalidate the entire preview
        - Tier 1 (Auto-stop): Quick edits handled by auto-stop on interaction
        - Tier 3 (Always allow): Non-destructive operations
        
        Args:
            enabled: True to enable editing, False to disable during preview
        """
        # Tier 2: DISABLE destructive operations during preview
        
        # Asset browser - prevent changing background/templates
        if hasattr(self, 'asset_browser'):
            self.asset_browser.background_folder_button.setEnabled(enabled)
            self.asset_browser.cell_line_combo.setEnabled(enabled)
            self.asset_browser.cell_template_list.setEnabled(enabled)
            
            # Set tooltips on disabled elements
            if not enabled:
                self.asset_browser.background_folder_button.setToolTip("Stop preview to change background")
                self.asset_browser.cell_line_combo.setToolTip("Stop preview to change cell line")
                self.asset_browser.cell_template_list.setToolTip("Stop preview to change cell templates")
            else:
                self.asset_browser.background_folder_button.setToolTip("")
                self.asset_browser.cell_line_combo.setToolTip("")
                self.asset_browser.cell_template_list.setToolTip("")
        
        # Cell inspector - prevent adding/removing cells
        if hasattr(self, 'cell_inspector'):
            # Note: Cell selection and trajectory editing will auto-stop preview (Tier 1)
            # We just disable the ability to see list interaction, but selection will auto-stop
            # The list items have individual remove buttons, but disabling selection prevents removal
            pass  # Cell selection auto-stop handled in signal handlers
        
        # Schedule widget - prevent editing frequency intervals
        if hasattr(self, 'schedule_widget'):
            self.schedule_widget.add_interval_button.setEnabled(enabled)
            self.schedule_widget.remove_interval_button.setEnabled(enabled)
            self.schedule_widget.schedule_table.setEnabled(enabled)
            
            if not enabled:
                self.schedule_widget.add_interval_button.setToolTip("Stop preview to add intervals")
                self.schedule_widget.remove_interval_button.setToolTip("Stop preview to remove intervals")
                self.schedule_widget.schedule_table.setToolTip("Stop preview to edit frequency intervals")
            else:
                self.schedule_widget.add_interval_button.setToolTip("")
                self.schedule_widget.remove_interval_button.setToolTip("")
                self.schedule_widget.schedule_table.setToolTip("")
        
        # Update status bar to show preview mode
        if not enabled:
            # In preview mode
            self.statusBar().showMessage("PREVIEW MODE - Editing restricted (Stop to modify scenario)", 0)
        else:
            # Editing enabled
            self.statusBar().clearMessage()

    def _auto_stop_preview_for_edit(self, edit_description: str = "trajectory edit") -> None:
        """Auto-stop preview when user attempts a Tier 1 (quick edit) operation.
        
        Args:
            edit_description: Description of the edit for status message
        """
        if self.preview_controller and self.preview_controller.is_active():
            self._preview_finish_message = f"Preview stopped ({edit_description})"
            self.preview_controller.stop()

    def toggle_preview_playback(self) -> None:
        """Legacy method for backward compatibility - now redirects to play/pause."""
        self._on_play_pause_clicked()
    
    def _start_preview(self) -> None:
        """Start a new preview animation (playing)."""
        self._start_preview_internal(start_playing=True)
    
    def _start_preview_paused(self) -> None:
        """Start preview in paused state at frame 0."""
        self._start_preview_internal(start_playing=False)
    
    def _start_preview_at_frame(self, frame: int) -> None:
        """Start preview paused at a specific frame.
        
        Args:
            frame: Frame number to start at
        """
        self._start_preview_internal(start_playing=False, start_frame=frame)
    
    def _start_preview_internal(self, start_playing: bool = True, start_frame: int = 0) -> None:
        """Internal method to start preview with various options.
        
        Args:
            start_playing: If True, starts playing immediately. If False, starts paused.
            start_frame: Frame to start at (default: 0)
        """
        if not self.preview_controller:
            return

        self._preview_finish_message = None
        payload = self._prepare_preview_payload()
        if payload is None:
            return

        background, templates, ground_truth, frame_count, idle_duration = payload
        if frame_count <= 0:
            QMessageBox.warning(self, "Preview", "Scenario produces zero frames; nothing to preview.")
            return

        if self._thumbnail_manager:
            self._thumbnail_manager.set_enabled(False)

        self._clear_handles()
        
        # Show playhead when entering preview mode
        if self.schedule_timeline:
            self.schedule_timeline.show_playhead()
        
        # Show overlay to lock left panel
        if hasattr(self, 'left_panel_overlay'):
            self.left_panel_overlay.show()
            self.left_panel_overlay._update_geometry()  # Ensure correct positioning
            self.left_panel_overlay.raise_()  # Ensure it's on top
        
        # Disable editing widgets during preview (Tier 2: Destructive changes)
        self._set_editing_enabled(False)
        
        # Update button state based on whether we're starting playing or paused
        if start_playing:
            self._update_button_state_playing()
        else:
            self._update_button_state_paused()

        self._preview_finish_message = "Preview finished"
        feather_params = self._edge_feather_runtime_params()
        self.preview_controller.start(
            background.frame_for(0),
            templates,
            ground_truth,
            frame_count,
            noise_enabled=self.state.noise_enabled,
            noise_stddev=self.state.noise_stddev,
            feather_params=feather_params,
            background_provider=background,
        )
        
        # If starting paused, pause immediately
        if not start_playing:
            self.preview_controller.pause()
        
        # If starting at a specific frame, seek to it
        if start_frame > 0:
            self.preview_controller.seek(start_frame)
            time_s = start_frame / ground_truth.fps if ground_truth.fps > 0 else 0.0
            self.schedule_timeline.set_playhead_position(start_frame, time_s)

        message = "Preview started"
        if not start_playing:
            message += " (paused)"
        if idle_duration:
            message += f" (idle interval {idle_duration:.3f}s appended)"
        self.statusBar().showMessage(message)

    def _on_preview_finished(self) -> None:
        self._update_button_state_stopped()
        
        # Reset playhead to home position without re-triggering preview activity
        if self.schedule_timeline:
            self.schedule_timeline.reset_to_start(ensure_visible=True)
            self._update_timeline_status_label(0, 0.0)
        
        # Hide overlay to unlock left panel
        if hasattr(self, 'left_panel_overlay'):
            self.left_panel_overlay.hide()
        
        # Re-enable editing widgets
        self._set_editing_enabled(True)
        
        # Set focus to the plot widget to prevent accidental Save Video trigger
        # This prevents Space key from triggering the Save Video button after clicking Stop
        if hasattr(self, 'plot_widget') and self.plot_widget:
            self.plot_widget.setFocus()
        
        if self._thumbnail_manager:
            self._thumbnail_manager.set_enabled(self._thumbnails_enabled)
        
        self._refresh_plot()
        message = self._preview_finish_message or "Preview finished"
        self.statusBar().showMessage(message)
        self._preview_finish_message = None
    
    def _update_control_states(self) -> None:
        """Update enabled/visible state of all controls based on scenario state.
        
        Implements progressive disclosure pattern:
        - State 1: No background → Cell selection enabled, canvas hidden, trajectory fields hidden
        - State 2: Background only → Canvas shown (with welcome if no cells), trajectory fields visible when cell selected
        - State 3: Background + Cells → Preview/export enabled, trajectories drawn on canvas
        """
        has_background = self.state.has_background_source()
        has_cells = len(self.state.cells) > 0
        has_schedule = len(self.state.schedule) > 0
        selected_idx = self._current_cell_index()
        has_selection = selected_idx is not None and 0 <= selected_idx < len(self.state.cells)
        
        # Progressive disclosure: Enable controls based on what's loaded
        
        # Save Config: Only enable if ALL essential elements are present
        has_complete_scenario = has_background and has_cells and has_schedule
        if hasattr(self, 'save_config_action'):
            self.save_config_action.setEnabled(has_complete_scenario)
            if not has_background:
                self.save_config_action.setStatusTip("Add a background source to enable save")
            elif not has_cells:
                self.save_config_action.setStatusTip("Add cells to scenario to enable save")
            elif not has_schedule:
                self.save_config_action.setStatusTip("Add schedule intervals to enable save")
            else:
                self.save_config_action.setStatusTip("Save current scenario configuration to file")
        
        # Cell line/template selection: Always enabled (no background required)
        if hasattr(self, 'cell_line_combo'):
            self.cell_line_combo.setEnabled(True)
            self.cell_line_combo.setToolTip("Select cell line to view templates")
        
        if hasattr(self, 'cell_template_list'):
            self.cell_template_list.setEnabled(True)
            self.cell_template_list.setToolTip("Double-click to add cell to scenario")
        
        # Play/Save Video require background, cells, AND schedule
        if hasattr(self, 'play_pause_button'):
            self.play_pause_button.setEnabled(has_background and has_cells and has_schedule)
            if not has_background:
                tooltip = "Select a background source first to enable preview"
            elif not has_cells:
                tooltip = "Add cells to scenario to enable preview"
            elif not has_schedule:
                tooltip = "Add schedule intervals to enable preview"
            else:
                tooltip = "Play preview animation (Shortcut: Space)"
            self.play_pause_button.setToolTip(tooltip)
            if hasattr(self, "play_pause_action"):
                self.play_pause_action.setEnabled(self.play_pause_button.isEnabled())
                self.play_pause_action.setStatusTip(tooltip)
        
        if hasattr(self, 'export_button'):
            self.export_button.button.setEnabled(has_background and has_cells and has_schedule)
            if not has_background:
                tooltip = "Select a background source first to enable Save Video"
            elif not has_cells:
                tooltip = "Add cells to scenario to enable Save Video"
            elif not has_schedule:
                tooltip = "Add schedule intervals to enable Save Video"
            else:
                tooltip = "Save video to file"
            self.export_button.button.setToolTip(tooltip)
            if hasattr(self, "save_video_action"):
                self.save_video_action.setEnabled(self.export_button.button.isEnabled())
                self.save_video_action.setStatusTip(tooltip)
        
        # Cell Inspector: Show header with cell info, hide trajectory fields until background + selection
        if hasattr(self, 'cell_inspector'):
            # Header with cell name/radius: visible when cell is selected
            header_visible = has_cells and has_selection
            
            # Trajectory controls: only show when background exists AND cell is selected
            trajectory_visible = has_background and has_cells and has_selection
            
            # Show/hide trajectory type selector and position fields
            self.cell_inspector.trajectory_combo.setVisible(trajectory_visible)
            self.cell_inspector.trajectory_label.setVisible(trajectory_visible)
            if hasattr(self.cell_inspector, "trajectory_row"):
                self.cell_inspector.trajectory_row.setVisible(trajectory_visible)
            if trajectory_visible:
                self.cell_inspector.start_label.setVisible(True)
                self.cell_inspector.start_x_spin.setVisible(True)
                self.cell_inspector.start_y_spin.setVisible(True)
                # Let _update_coordinate_fields_from_cell manage end/control visibility
                if has_selection and selected_idx is not None and 0 <= selected_idx < len(self.state.cells):
                    self._update_coordinate_fields_from_cell(self.state.cells[selected_idx])
            else:
                self.cell_inspector.start_label.setVisible(False)
                self.cell_inspector.start_x_spin.setVisible(False)
                self.cell_inspector.start_y_spin.setVisible(False)
                self.cell_inspector.end_label.setVisible(False)
                self.cell_inspector.end_x_spin.setVisible(False)
                self.cell_inspector.end_y_spin.setVisible(False)
                if hasattr(self.cell_inspector, "control1_label"):
                    self.cell_inspector.control1_label.setVisible(False)
                    self.cell_inspector.control1_x_spin.setVisible(False)
                    self.cell_inspector.control1_y_spin.setVisible(False)
                if hasattr(self.cell_inspector, "control2_label"):
                    self.cell_inspector.control2_label.setVisible(False)
                    self.cell_inspector.control2_x_spin.setVisible(False)
                    self.cell_inspector.control2_y_spin.setVisible(False)
        
        # Background warnings based on current selection
        self._refresh_background_warning()

        # Update status bar with contextual hint
        self._update_status_hint(has_background, has_cells, has_schedule)

        if self.schedule_timeline:
            self.schedule_timeline.set_interaction_enabled(has_complete_scenario)
    
    def _update_status_hint(self, has_background: bool, has_cells: bool, has_schedule: bool) -> None:
        """Update status bar with contextual guidance for user."""
        # Don't override status messages during operations
        if hasattr(self, 'preview_controller') and self.preview_controller and self.preview_controller.is_active():
            return  # Keep preview status messages
        
        if not has_background:
            self.statusBar().showMessage("Select a background from the left panel to begin")
        elif not has_cells:
            self.statusBar().showMessage("Add cells by double-clicking templates")
        elif not has_schedule:
            self.statusBar().showMessage("Add schedule intervals to define cell behavior over time")
        else:
            self.statusBar().showMessage("Ready — Click Play to preview or Save Video to render")

    def _estimate_required_frames(self) -> int:
        total_frames = 0
        for interval in self.state.schedule:
            total_frames += math.ceil(interval.duration_s * self.state.fps)
            total_frames += math.ceil(interval.delay_after_s * self.state.fps)
        return total_frames

    def _refresh_background_warning(self) -> None:
        warning_label = getattr(self.asset_browser, "background_warning", None)
        if warning_label is None:
            return

        if self.state.background_frames_dir is None or not self.state.background_frames_dir.exists():
            warning_label.hide()
            return

        from ..assets import _list_frame_files

        frame_paths = _list_frame_files(self.state.background_frames_dir)
        if not frame_paths:
            warning_label.setText("⚠️ No image frames found in the selected folder.")
            warning_label.show()
            return

        required_frames = self._estimate_required_frames()
        if required_frames <= 0:
            warning_label.hide()
            return

        if len(frame_paths) < required_frames:
            if len(frame_paths) <= 2:
                warning_label.setText(
                    f"⚠️ Background frames: {len(frame_paths)} < required {required_frames}. "
                    "The same frame(s) will repeat."
                )
            else:
                warning_label.setText(
                    f"⚠️ Background frames: {len(frame_paths)} < required {required_frames}. "
                    "Auto-reverse looping will be applied."
                )
            warning_label.show()
        else:
            warning_label.hide()

    def _update_background_summary(self) -> None:
        summary_label = getattr(self.asset_browser, "background_summary", None)
        set_thumb = getattr(self.asset_browser, "set_background_thumbnail", None)
        folder_button = getattr(self.asset_browser, "background_folder_button", None)
        if summary_label is None or set_thumb is None:
            return

        # Prefer frame directory when available; fall back to static background image
        background_dir = self.state.background_frames_dir
        background_image = self.state.background if background_dir is None else None

        if folder_button:
            tooltip_target = background_dir or background_image
            folder_button.setToolTip(str(tooltip_target) if tooltip_target else "")

        if background_dir:
            if not background_dir.exists():
                summary_label.setText(f"Folder: {background_dir}\n(Missing on disk)")
                set_thumb(None)
                return

            from ..assets import _list_frame_files

            frame_paths = _list_frame_files(background_dir)
            if not frame_paths:
                summary_label.setText("No frames found in selected folder")
                set_thumb(None)
                return

            first = frame_paths[0]
            image = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
            if image is None:
                summary_label.setText(f"{background_dir.name}: failed to load preview")
                set_thumb(None)
                return

            h, w = image.shape
            summary_label.setText(
                f"Folder: {background_dir.name}\n"
                f"Frames: {len(frame_paths)}\n"
                f"Resolution: {w}×{h}"
            )

            qimage = QImage(image.data, w, h, QImage.Format.Format_Grayscale8)
            set_thumb(QPixmap.fromImage(qimage.copy()))
            return

        if background_image:
            if not background_image.exists():
                summary_label.setText(f"Image: {background_image}\n(Missing on disk)")
                set_thumb(None)
                return

            image = cv2.imread(str(background_image), cv2.IMREAD_GRAYSCALE)
            if image is None:
                summary_label.setText(f"{background_image.name}: failed to load preview")
                set_thumb(None)
                return

            h, w = image.shape
            summary_label.setText(
                f"Image: {background_image.name}\n"
                f"Resolution: {w}×{h}"
            )
            qimage = QImage(image.data, w, h, QImage.Format.Format_Grayscale8)
            set_thumb(QPixmap.fromImage(qimage.copy()))
            return

        summary_label.setText("No folder selected")
        set_thumb(None)
    
    def _create_welcome_widget(self) -> QWidget:
        """Create napari-style welcome widget shown when no background is loaded."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Add top stretch for vertical centering
        layout.addStretch(1)
        
        # SVG icon (large, centered, tinted light gray)
        icon_path = Path(__file__).parent / "microscope_icon.svg"
        if icon_path.exists():
            size = QSize(210, 210)  # Double size - prominent like napari
            pixmap = QPixmap(size)
            pixmap.fill(Qt.GlobalColor.transparent)
            renderer = QSvgRenderer(str(icon_path))
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.setCompositionMode(QPainter.CompositionMode_SourceIn)  # type: ignore[attr-defined]
            painter.fillRect(pixmap.rect(), QColor(60, 60, 60))  # Gray tint
            painter.end()

            icon_label = QLabel()
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setPixmap(pixmap)
            layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Main message
        title_label = QLabel("No Background Loaded")
        title_font = title_label.font()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Instructions (subtle, like napari's help text)
        instruction_label = QLabel("Select a background from the left panel to begin")
        instruction_font = instruction_label.font()
        instruction_font.setPointSize(12)
        instruction_label.setFont(instruction_font)
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instruction_label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        layout.addWidget(instruction_label)
        
        # Add bottom stretch for vertical centering
        layout.addStretch(1)
        
        # Set background color (subtle gray like napari)
        widget.setStyleSheet("background-color: rgba(50, 50, 50, 1);")
        
        return widget
    
    def _show_welcome_widget(self) -> None:
        """Show welcome widget and hide canvas."""
        if self._welcome_widget:
            self._welcome_widget.show()
        self.plot_widget.hide()
    
    def _hide_welcome_widget(self) -> None:
        """Hide welcome widget and show canvas."""
        if self._welcome_widget:
            self._welcome_widget.hide()
        self.plot_widget.show()
    
    def _initialize_empty_timeline(self) -> None:
        """Initialize timeline with sensible defaults when no schedule exists."""
        # Set default 10-second duration at 30 fps
        default_duration = 10.0
        default_fps = 30.0
        
        plot = self.schedule_timeline.getPlotItem()
        plot.setLimits(xMin=0, xMax=default_duration)
        plot.setXRange(0, default_duration)
        plot.setYRange(-0.5, 0.5, padding=0)
        
        # Update internal timeline state
        self.schedule_timeline._fps = default_fps
        self.schedule_timeline._total_frames = int(default_duration * default_fps)
        
        self._update_timeline_status_label(0, 0.0)

    def _format_timecode(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        secs = int(round(seconds - minutes * 60))
        if secs == 60:
            minutes += 1
            secs = 0
        return f"{minutes}:{secs:02d}"

    def _update_timeline_status_label(self, frame: Optional[int] = None, time_s: Optional[float] = None) -> None:
        if not hasattr(self, "timeline_status_label") or self.timeline_status_label is None:
            return
        if not hasattr(self, "schedule_timeline") or self.schedule_timeline is None:
            self.timeline_status_label.setText("0:00 / 0:00 • Frame 0 / 0")
            return
        if hasattr(self, "state") and (not self.state.schedule):
            self.timeline_status_label.setText("0:00 / 0:00 • Frame 0 / 0")
            return
        total_frames = max(0, int(getattr(self.schedule_timeline, "_total_frames", 0)))
        fps = float(getattr(self.schedule_timeline, "_fps", 0.0))
        if fps <= 0:
            fps = float(self.state.fps) if hasattr(self, "state") else 0.0
        if total_frames <= 0 or fps <= 0:
            self.timeline_status_label.setText("0:00 / 0:00 • Frame 0 / 0")
            return
        if frame is None:
            frame = int(getattr(self.schedule_timeline, "_current_frame", 0))
        frame = max(0, min(frame, total_frames - 1))
        if time_s is None:
            time_s = frame / fps if fps > 0 else 0.0
        total_time = total_frames / fps if fps > 0 else 0.0
        display_frame = min(total_frames, frame + 1)
        label = f"{self._format_timecode(time_s)} / {self._format_timecode(total_time)} • Frame {display_frame} / {total_frames}"
        self.timeline_status_label.setText(label)
    
    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts for preview control.
        
        Shortcuts:
        - Space: Start preview or Play/Pause (if already in preview)
        - Esc: Stop/quit preview mode
        - Left Arrow: Step backward one frame (only when preview is active)
        - Right Arrow: Step forward one frame (only when preview is active)
        """
        key = event.key()
        
        # ESC: Stop preview (if active)
        if key == Qt.Key.Key_Escape:
            if self.preview_controller and self.preview_controller.is_active():
                self._on_stop_clicked()
                event.accept()
                return
            else:
                super().keyPressEvent(event)
                return
        
        # Space: Start preview or toggle Play/Pause
        if key == Qt.Key.Key_Space:
            # Check if a text input widget has focus (prevent interference with typing)
            from PySide6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox
            focused_widget = QApplication.focusWidget()
            if isinstance(focused_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)):
                # Don't handle Space - let the widget handle it normally
                super().keyPressEvent(event)
                return
            
            # Check if Play button is enabled (same requirements as button)
            has_background = self.state.has_background_source()
            has_cells = len(self.state.cells) > 0
            has_schedule = len(self.state.schedule) > 0
            can_play = has_background and has_cells and has_schedule
            
            if not can_play:
                # Play button is disabled - Space shortcut should also be disabled
                super().keyPressEvent(event)
                return
            
            if self.preview_controller and self.preview_controller.is_active():
                # Already in preview: toggle Play/Pause
                self._on_play_pause_clicked()
            else:
                # Not in preview: start preview playing (same as clicking Play button)
                self._start_preview()
            event.accept()
            return
        
        # Arrow keys: Only work when preview is active
        if not self.preview_controller or not self.preview_controller.is_active():
            super().keyPressEvent(event)
            return
        
        # Check if arrow keys - need special handling for spinboxes
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            # Check if a spinbox has focus (they use arrows for increment/decrement)
            from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox
            focused_widget = QApplication.focusWidget()
            if isinstance(focused_widget, (QSpinBox, QDoubleSpinBox)):
                # Let the spinbox handle arrow keys normally
                super().keyPressEvent(event)
                return
            
            # No spinbox focused - use arrows for frame stepping
            if key == Qt.Key.Key_Left:
                # Left Arrow: Step backward one frame
                self._step_frame(-1)
                event.accept()
            elif key == Qt.Key.Key_Right:
                # Right Arrow: Step forward one frame
                self._step_frame(1)
                event.accept()
        else:
            super().keyPressEvent(event)
    
    def _step_frame(self, delta: int) -> None:
        """Step forward or backward by a specific number of frames.
        
        Args:
            delta: Number of frames to step (positive = forward, negative = backward)
        """
        if not self.preview_controller or not self.preview_controller.is_active():
            return
        
        # Get current frame from timeline
        current_frame = self.schedule_timeline._current_frame
        
        # Calculate new frame
        new_frame = current_frame + delta
        
        # Clamp to valid range
        total_frames = self.schedule_timeline._total_frames
        if new_frame < 0:
            new_frame = 0
        elif new_frame >= total_frames:
            new_frame = total_frames - 1
        
        # Pause if playing
        if self.preview_controller._timer.isActive():
            self.preview_controller.pause()
            self._update_button_state_paused()
        
        # Seek to new frame
        time_s = new_frame / self.schedule_timeline._fps if self.schedule_timeline._fps > 0 else 0.0
        self.preview_controller.seek(new_frame)
        self.schedule_timeline.set_playhead_position(new_frame, time_s)
        
        # Update status
        direction = "forward" if delta > 0 else "backward"
        self.statusBar().showMessage(f"Stepped {direction} to frame {new_frame} ({time_s:.2f}s)")

    def _prepare_preview_payload(
        self,
    ) -> Optional[
        Tuple[
            object,
            List[CellTemplate],
            GroundTruth,
            int,
            Optional[float],
        ]
    ]:
        try:
            config, idle_duration = self.state.to_config()
            assets = AssetManager(config)
            background = assets.load_background()
            templates = assets.load_cell_templates()
            schedule = build_frame_schedule(config)
            renderer = Renderer(config, background, templates, schedule)
            ground_truth = renderer.compute_ground_truth()
            frame_count = schedule.total_frames
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preview", str(exc))
            return None

        return background, templates, ground_truth, frame_count, idle_duration

    def render_current_scenario(self) -> None:
        if self.preview_controller and self.preview_controller.is_active():
            self._preview_finish_message = "Preview stopped"
            self.preview_controller.stop()

        default_dir = default_output_root()
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_path = default_dir / f"{default_name}.avi"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video As",
            str(suggested_path),
            "AVI Video (*.avi)",
        )
        if not file_path:
            return
        chosen_path = Path(file_path)
        if not chosen_path.suffix:
            chosen_path = chosen_path.with_suffix(".avi")
        output_root = chosen_path.parent
        scenario_name_override = chosen_path.stem

        try:
            config, idle_duration = self.state.to_config()
            assets = AssetManager(config)
            background = assets.load_background()
            templates = assets.load_cell_templates()
            schedule = build_frame_schedule(config)
            renderer = Renderer(config, background, templates, schedule)
            feather_params = self._edge_feather_runtime_params()
            frames, ground_truth = renderer.render(
                feather_params=feather_params,
            )
            output_dir = export_simulation_outputs(
                frames,
                ground_truth,
                config,
                output_root=output_root,
                feather_params=feather_params,
                feather_pixels=0,
                scenario_name=scenario_name_override,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Render", str(exc))
            return

        text = f"Outputs written to {output_dir}"
        if idle_duration:
            text += f". Idle interval {idle_duration:.3f}s appended to match video length."
        QMessageBox.information(self, "Render", text)
        self.statusBar().showMessage(text)

    # Config IO -----------------------------------------------------------
    def open_config_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Simulation Config",
            str(self._examples_dir),
            "YAML Files (*.yaml *.yml)",
        )
        if file_path:
            self.load_config(Path(file_path))

    def load_config(self, path: Path) -> None:
        try:
            config = load_simulation_config(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open", f"Failed to load config: {exc}")
            return

        def _points_from_config(traj: TrajectoryConfig) -> Tuple[Tuple[float, float], Tuple[float, float]]:
            if traj.type == "stationary":
                position = traj.position or traj.start or traj.end
                if position is None:
                    raise ValueError("Stationary trajectory missing 'position'")
                pt = tuple(position)
                return pt, pt
            if traj.start is None or traj.end is None:
                raise ValueError(f"{traj.type} trajectory requires 'start' and 'end'")
            return tuple(traj.start), tuple(traj.end)

        scenario_cells: List[ScenarioCell] = []
        for cell in config.cells:
            start_pt, end_pt = _points_from_config(cell.trajectory)
            scenario_cells.append(
                ScenarioCell(
                    id=cell.id,
                    template=cell.template,
                    mask=cell.mask.path,
                    trajectory_type=cell.trajectory.type,
                    start=start_pt,
                    end=end_pt,
                    control_points=list(cell.trajectory.control_points or []),
                )
            )

        self.state = ScenarioState(
            background=config.video.background,
            background_frames_dir=config.video.background_frames_dir,
            noise_enabled=config.video.noise_enabled,
            noise_stddev=config.video.noise_stddev,
            cells=scenario_cells,
            schedule=[
                ScheduleInterval(
                    frequency_khz=interval.frequency_khz,
                    duration_s=interval.duration_s,
                    angular_velocity_rad_s=interval.angular_velocity_rad_s or 0.0,
                    delay_after_s=interval.delay_after_s or 0.0,
                )
                for interval in config.schedule.intervals
            ],
            resolution=tuple(config.video.resolution),
            fps=config.video.fps,
            magnification=config.video.magnification,
            pixel_size_um=config.video.pixel_size_um,
            video_duration_s=config.video.duration_s,
            background_ref_mag=config.video.background_ref_mag,
        )
        edge_meta = config.metadata.get("edge_feather")
        if isinstance(edge_meta, dict):
            try:
                self.state.edge_feather.enabled = bool(edge_meta.get("enabled", False))
                self.state.edge_feather.inside_pixels = float(edge_meta.get("inside_pixels", self.state.edge_feather.inside_pixels))
                self.state.edge_feather.outside_pixels = float(edge_meta.get("outside_pixels", self.state.edge_feather.outside_pixels))
            except Exception:
                pass
        for cell in self.state.cells:
            cell.ensure_controls(self.state.resolution)
        self.statusBar().showMessage(f"Loaded {path}")
        # Clear selection state before loading new scenario to ensure clean initialization
        self.selection_state.clear()
        self._reload_asset_lists()
        default_select_id = self.state.cells[0].id if self.state.cells else None
        self._refresh_scenario_list(select_id=default_select_id)
        self._refresh_schedule_table()
        self._sync_video_settings_from_state()
        self._refresh_plot()
        self._update_background_summary()
        self._update_control_states()  # Update UI state

    def save_config_dialog(self) -> None:
        """Save current GUI state as YAML config file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simulation Config",
            str((self._examples_dir / "config.yaml")),
            "YAML Files (*.yaml *.yml)",
        )
        if not file_path:
            return
        
        try:
            config, _ = self.state.to_config()
            # Write YAML with minimal metadata
            import yaml
            from datetime import datetime

            def to_serializable(value):
                """Recursively convert values to YAML-friendly types."""
                if isinstance(value, dict):
                    return {key: to_serializable(val) for key, val in value.items()}
                if isinstance(value, tuple):
                    return [to_serializable(item) for item in value]
                if isinstance(value, list):
                    return [to_serializable(item) for item in value]
                if isinstance(value, Path):
                    return str(value)
                if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
                    try:
                        return value.tolist()
                    except Exception:  # noqa: BLE001
                        return value
                if hasattr(value, "item") and callable(getattr(value, "item")):
                    try:
                        return value.item()
                    except Exception:  # noqa: BLE001
                        return value
                return value

            def prune_none(mapping: dict) -> dict:
                return {k: v for k, v in mapping.items() if v is not None}

            video_dict = config.video.dict()
            video_dict = prune_none(video_dict)

            payload = to_serializable(
                {
                    "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video": video_dict,
                    "schedule": config.schedule.dict(),
                    "cells": [cell.dict() for cell in config.cells],
                    "metadata": config.metadata,
                }
            )

            output_path = Path(file_path)
            with output_path.open("w") as f:
                yaml.safe_dump(
                    payload,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            self.statusBar().showMessage(f"Saved config to {output_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Config", f"Failed to save config: {exc}")

    def _load_schedule_from_yaml(self) -> None:
        """Load schedule intervals from standalone YAML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Schedule",
            str(self._examples_dir),
            "YAML Files (*.yaml *.yml)",
        )
        if not file_path:
            return
        
        try:
            import yaml
            
            with Path(file_path).open("r") as f:
                data = yaml.safe_load(f)
            
            # Support both full config and standalone schedule
            if "schedule" in data:
                schedule_data = data["schedule"]
            else:
                schedule_data = data
            
            # Parse intervals
            if "intervals" not in schedule_data:
                raise ValueError("YAML must contain 'intervals' list")
            
            intervals = []
            for interval_dict in schedule_data["intervals"]:
                intervals.append(
                    ScheduleInterval(
                        frequency_khz=float(interval_dict["frequency_khz"]),
                        duration_s=float(interval_dict["duration_s"]),
                        angular_velocity_rad_s=float(interval_dict.get("angular_velocity_rad_s", 0.0)),
                        delay_after_s=float(interval_dict.get("delay_after_s", 0.0)),
                    )
                )
            
            # Update state
            self.state.schedule = intervals
            # Auto-compute video duration from schedule
            self.state.video_duration_s = sum(i.duration_s + i.delay_after_s for i in intervals)
            self.statusBar().showMessage(f"Loaded schedule from {file_path}")
            self._update_video_info_label()
            self.schedule_timeline.update_from_state(self.state)
            self._update_timeline_status_label()
            self._refresh_schedule_table()
            self._update_control_states()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Schedule", f"Failed to load schedule: {exc}")

    def _sync_video_settings_from_state(self) -> None:
        """Sync video settings spinboxes from state without triggering change events."""
        # Block signals to prevent recursive updates
        self.resolution_width_spin.blockSignals(True)
        self.resolution_height_spin.blockSignals(True)
        self.fps_spin.blockSignals(True)
        self.magnification_spin.blockSignals(True)
        self.bg_ref_mag_spin.blockSignals(True)
        self.pixel_size_spin.blockSignals(True)
        self.noise_enabled_check.blockSignals(True)
        self.noise_stddev_spin.blockSignals(True)
        
        # Update values
        self.resolution_width_spin.setValue(float(self.state.resolution[0]))
        self.resolution_height_spin.setValue(float(self.state.resolution[1]))
        self.fps_spin.setValue(self.state.fps)
        self.magnification_spin.setValue(self.state.magnification)
        self.bg_ref_mag_spin.setValue(self.state.background_ref_mag)
        self.pixel_size_spin.setValue(self.state.pixel_size_um)
        self.noise_enabled_check.setChecked(self.state.noise_enabled)
        self._update_noise_button_text()
        self.noise_stddev_spin.setValue(self.state.noise_stddev)
        self._update_noise_controls_enabled()
        
        # Unblock signals
        self.resolution_width_spin.blockSignals(False)
        self.resolution_height_spin.blockSignals(False)
        self.fps_spin.blockSignals(False)
        self.magnification_spin.blockSignals(False)
        self.bg_ref_mag_spin.blockSignals(False)
        self.pixel_size_spin.blockSignals(False)
        self.noise_enabled_check.blockSignals(False)
        self.noise_stddev_spin.blockSignals(False)
        
        # Update warning and info label
        self._update_magnification_warning()
        self._update_video_info_label()
        self._sync_edge_feather_controls_from_state()

    def _unique_cell_id(self, base: str) -> str:
        base = base.replace(" ", "_")
        existing = {cell.id for cell in self.state.cells}
        if base not in existing:
            return base
        counter = 1
        while True:
            candidate = f"{base}_{counter:02d}"
            if candidate not in existing:
                return candidate
            counter += 1

    def _cell_line_for_template(self, template: Path) -> Optional[str]:
        parts = template.parts
        if "cell_lines" in parts:
            idx = parts.index("cell_lines")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return None

    def _refresh_plot(self) -> None:
        """Refresh the plot canvas with background, trajectories, and interactive handles."""
        self.plot_item.clear()
        self._active_curve = None
        self._trajectory_items.clear()
        self._clear_handles()
        feather_params = self._edge_feather_runtime_params()

        # Show empty state widget if no background loaded
        if not self.state.has_background_source():
            self._show_welcome_widget()
            self._background_extent = (self.state.resolution[0], self.state.resolution[1])
            if self._thumbnail_manager:
                self._thumbnail_manager.clear()
            self._update_timeline_status_label()
            return

        # Hide welcome widget and show canvas
        self._hide_welcome_widget()

        # Draw background with magnification zoom (cropping/padding as needed)
        from ..assets import _scale_background, _list_frame_files

        data: Optional[np.ndarray] = None
        if self.state.background_frames_dir:
            frame_paths = _list_frame_files(self.state.background_frames_dir)
            if frame_paths:
                data = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
        elif self.state.background:
            data = cv2.imread(str(self.state.background), cv2.IMREAD_GRAYSCALE)

        if data is None:
            self._show_welcome_widget()
            self._background_extent = (self.state.resolution[0], self.state.resolution[1])
            if self._thumbnail_manager:
                self._thumbnail_manager.clear()
            self._update_timeline_status_label()
            return

        # Apply zoom based on magnification vs reference magnification
        target_width, target_height = self.state.resolution
        zoomed_bg = _scale_background(
            data,
            (target_width, target_height),
            self.state.magnification,
            self.state.background_ref_mag
        )
        
        # Create ImageItem with fixed levels to prevent auto-normalization
        # This ensures the preview matches the exported video exactly
        bg_item = pg.ImageItem(zoomed_bg.T)
        bg_item.setLevels([0, 255])  # Lock to full uint8 range, no auto-scaling
        self.plot_item.addItem(bg_item)
        self._background_extent = (zoomed_bg.shape[1], zoomed_bg.shape[0])

        # Get currently selected cell index
        selected_idx = self._current_cell_index()
        thumbnail_selected = selected_idx if 0 <= selected_idx < len(self.state.cells) else None

        if self._thumbnail_manager:
            self._thumbnail_manager.rebuild(
                self.state.cells,
                thumbnail_selected,
                self.state.magnification,
                self.state.pixel_size_um,
                feather_params,
                self.state.noise_stddev if self.state.noise_enabled else None,
            )
        self._update_timeline_status_label()
        
        for idx, cell in enumerate(self.state.cells):
            path = self._trajectory_path(cell, samples=200)
            if path is not None and len(path) > 1:
                x_data, y_data = path[:, 0], path[:, 1]
            else:
                x_data = [cell.start[0], cell.end[0]]
                y_data = [cell.start[1], cell.end[1]]

            if idx == selected_idx:
                curve = TrajectoryItem(
                    idx,
                    self,
                    self.selection_state,
                    x_data,
                    y_data,
                )
                curve.setZValue(30)
                self._active_curve = curve
            else:
                curve = SelectableTrajectory(
                    idx,
                    self,
                    self.selection_state,
                    x_data,
                    y_data,
                )
                curve.setZValue(10)
                self._inactive_trajectories[idx] = curve
                self._create_inactive_markers(idx)

            self.plot_item.addItem(curve)
            self._trajectory_items[idx] = curve
        
        # Create handles for the selected cell (visible handles for interaction)
        if selected_idx is not None and 0 <= selected_idx < len(self.state.cells):
            self._attach_handles_for_cell(selected_idx)
        self._current_primary = selected_idx if selected_idx is not None and 0 <= selected_idx < len(self.state.cells) else None

    def _fit_view(self) -> None:
        if not hasattr(self, "plot_item") or self.plot_item is None:
            return
        vb = self.plot_item.vb
        if self._background_extent:
            width, height = self._background_extent
            vb.setRange(xRange=(0, width), yRange=(0, height), padding=0.02)
        else:
            vb.autoRange()
    
class QActionButton:
    def __init__(self, text: str, callback):
        self.button = QPushButton(text)
        self.button.clicked.connect(callback)


def run(initial_config: Optional[Path] = None) -> None:
    app = QApplication.instance() or QApplication([])
    
    # Force dark mode regardless of system settings
    app.setStyle("Fusion")
    from PySide6.QtGui import QPalette, QColor
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
    app.setPalette(dark_palette)
    
    editor = SimulatorEditor(initial_config)
    editor.show()
    app.exec()
