"""Dialog scaffolding for the upcoming mask editor tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, Signal, QSize, QRectF
from PySide6.QtGui import (
    QCloseEvent,
    QMouseEvent,
    QPen,
    QKeySequence,
    QShortcut,
    QIcon,
    QPixmap,
    QPainter,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QGraphicsEllipseItem,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QSizePolicy,
)

logger = logging.getLogger(__name__)


class MaskEditorDialog(QDialog):
    """Lightweight dialog shell that will host the mask editing workflow."""

    mask_saved = Signal(Path)

    def __init__(
        self,
        image_path: Path | str,
        mask_path: Path | str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)

        self._brush_radius = 2
        self._active_tool = "eraser"
        self._image_array: Optional[np.ndarray] = None
        self._mask_array: Optional[np.ndarray] = None
        self._mask_lut = self._build_mask_lut()
        self._mask_dirty = False
        self._image_dirty = False
        self._is_painting = False
        self._last_paint_pos: Optional[tuple[float, float]] = None
        self._cursor_view_pos: Optional[tuple[float, float]] = None
        self._crop_roi: Optional[pg.ROI] = None
        self._pending_crop_rect: Optional[tuple[int, int, int, int]] = None
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._max_history = 50
        self._stroke_changed = False
        self._overlay_opacity = 0.40
        self._icon_dir = Path(__file__).resolve().parents[1] / "icons"
        self._icon_cache: dict[str, QIcon] = {}
        self._pixmap_cache: dict[str, QPixmap] = {}

        self.setWindowTitle("Mask Editor")
        self.resize(900, 700)

        self._setup_ui()
        self._connect_signals()
        self._load_assets()

    # --------------------------------------------------------------------- UI
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        toolbar = self._build_toolbar()
        layout.addWidget(toolbar)

        self.canvas_widget = pg.GraphicsLayoutWidget()
        self.canvas_plot = self.canvas_widget.addPlot()
        self.canvas_plot.hideButtons()
        self.canvas_plot.setMenuEnabled(False)
        self.canvas_plot.setLabels(left="", bottom="")
        self.canvas_plot.showGrid(x=False, y=False)
        self.canvas_plot.setAspectLocked(True)
        self.canvas_plot.invertY(True)
        self.view_box = self.canvas_plot.getViewBox()
        self.view_box.setMouseEnabled(x=True, y=True)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.mask_item = pg.ImageItem(axisOrder="row-major")
        self.mask_item.setLookupTable(self._mask_lut)
        self.mask_item.setLevels([0, 255])
        self.mask_item.setOpacity(self._overlay_opacity)
        self.canvas_plot.addItem(self.image_item)
        self.canvas_plot.addItem(self.mask_item)

        self.cursor_item = QGraphicsEllipseItem()
        pen = QPen(Qt.white)
        pen.setWidthF(1.25)
        pen.setCosmetic(True)
        self.cursor_item.setPen(pen)
        self.cursor_item.setBrush(Qt.BrushStyle.NoBrush)
        self.cursor_item.setZValue(10)
        self.cursor_item.setVisible(False)
        self.view_box.addItem(self.cursor_item)

        self.canvas_viewport = self.canvas_widget.viewport()
        self.canvas_viewport.setMouseTracking(True)
        self.canvas_viewport.installEventFilter(self)

        layout.addWidget(self.canvas_widget, 1)

        self.status_label = QLabel("Load an image + mask to begin editing.")
        self.status_label.setObjectName("maskEditorStatus")
        layout.addWidget(self.status_label)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.save_button = self.button_box.button(QDialogButtonBox.StandardButton.Save)
        if self.save_button:
            icon = self._load_icon("save")
            if icon:
                self.save_button.setIcon(icon)
        layout.addWidget(self.button_box)

    def _build_toolbar(self) -> QWidget:
        toolbar = QWidget(self)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)

        self.move_button = QToolButton(toolbar)
        self.move_button.setText("Move")
        self.move_button.setCheckable(True)
        self.move_button.setToolTip("Pan the image")
        self._configure_tool_button(self.move_button)
        self._apply_icon(self.move_button, "drag_pan")

        self.crop_button = QToolButton(toolbar)
        self.crop_button.setText("Crop")
        self.crop_button.setCheckable(True)
        self.crop_button.setToolTip("Crop image + mask (Enter to apply, Esc to cancel)")
        self._configure_tool_button(self.crop_button)
        self._apply_icon(self.crop_button, "crop")

        self.brush_button = QToolButton(toolbar)
        self.brush_button.setText("Brush")
        self.brush_button.setCheckable(True)
        self.brush_button.setToolTip("Paint foreground pixels (shortcut: B)")
        self._configure_tool_button(self.brush_button)
        self._apply_icon(self.brush_button, "brush")

        self.eraser_button = QToolButton(toolbar)
        self.eraser_button.setText("Eraser")
        self.eraser_button.setCheckable(True)
        self.eraser_button.setChecked(True)
        self.eraser_button.setToolTip("Erase to background (shortcut: E)")
        self._configure_tool_button(self.eraser_button)
        self._apply_icon(self.eraser_button, "eraser")

        self.tool_group = QButtonGroup(toolbar)
        self.tool_group.setExclusive(True)
        self.tool_group.addButton(self.move_button)
        self.tool_group.addButton(self.crop_button)
        self.tool_group.addButton(self.brush_button)
        self.tool_group.addButton(self.eraser_button)

        toolbar_layout.addWidget(self.move_button)
        toolbar_layout.addWidget(self.crop_button)
        toolbar_layout.addWidget(self.brush_button)
        toolbar_layout.addWidget(self.eraser_button)

        toolbar_layout.addSpacing(12)

        size_label = self._make_icon_label("brush_size", "Brush Size:", toolbar)
        toolbar_layout.addWidget(size_label)

        self.size_slider = QSlider(Qt.Orientation.Horizontal, toolbar)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(20)
        self.size_slider.setPageStep(1)
        self.size_slider.setValue(self._brush_radius)
        self.size_slider.setToolTip("Adjust brush/eraser radius")
        toolbar_layout.addWidget(self.size_slider, 1)

        self.size_spin = QSpinBox(toolbar)
        self.size_spin.setMinimum(self.size_slider.minimum())
        self.size_spin.setMaximum(self.size_slider.maximum())
        self.size_spin.setValue(self._brush_radius)
        toolbar_layout.addWidget(self.size_spin)

        opacity_label = self._make_icon_label("opacity", "Opacity:", toolbar)
        toolbar_layout.addWidget(opacity_label)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, toolbar)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self._overlay_opacity * 100))
        self.opacity_slider.setToolTip("Mask overlay opacity")
        toolbar_layout.addWidget(self.opacity_slider, 1)
        self.opacity_value_label = QLabel(f"{int(self._overlay_opacity * 100)}%", toolbar)
        self.opacity_value_label.setFixedWidth(40)
        toolbar_layout.addWidget(self.opacity_value_label)

        toolbar_layout.addSpacing(12)

        self.undo_button = QToolButton(toolbar)
        self.undo_button.setText("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("Undo last stroke (Ctrl/Cmd+Z)")
        self._configure_tool_button(self.undo_button)
        self._apply_icon(self.undo_button, "undo")
        toolbar_layout.addWidget(self.undo_button)

        self.redo_button = QToolButton(toolbar)
        self.redo_button.setText("Redo")
        self.redo_button.setEnabled(False)
        self.redo_button.setToolTip("Redo stroke (Ctrl/Cmd+Shift+Z)")
        self._configure_tool_button(self.redo_button)
        self._apply_icon(self.redo_button, "redo")
        toolbar_layout.addWidget(self.redo_button)

        toolbar_layout.addStretch(0)
        return toolbar

    # ---------------------------------------------------------------- Signals
    def _connect_signals(self) -> None:
        self.move_button.toggled.connect(
            lambda checked: checked and self._set_active_tool("move")
        )
        self.crop_button.toggled.connect(
            lambda checked: checked and self._set_active_tool("crop")
        )
        self.brush_button.toggled.connect(
            lambda checked: checked and self._set_active_tool("brush")
        )
        self.eraser_button.toggled.connect(
            lambda checked: checked and self._set_active_tool("eraser")
        )
        self.size_slider.valueChanged.connect(self._on_brush_size_changed)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)

        self.undo_button.clicked.connect(self._undo_action)
        self.redo_button.clicked.connect(self._redo_action)

        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self._undo_action)
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self._redo_action)
        self._shortcuts = [undo_shortcut, redo_shortcut]

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self._handle_cancel)

    # ---------------------------------------------------------- Event hooks
    def _build_mask_lut(self) -> np.ndarray:
        """Create a LUT where 0 is transparent and 255 is semi-transparent coral."""
        lut = np.zeros((256, 4), dtype=np.ubyte)
        lut[255] = np.array([255, 64, 64, 180], dtype=np.ubyte)
        return lut

    def _apply_icon(self, button: QToolButton, name: str) -> None:
        icon = self._load_icon(name)
        if icon is not None:
            self._ensure_disabled_icon(icon)
            button.setIcon(icon)
        button.setIconSize(QSize(18, 18))

    def _make_icon_label(self, icon_name: str, text: str, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        icon_label = QLabel(container)
        pixmap = self._load_pixmap(icon_name, 16)
        if pixmap:
            icon_label.setPixmap(pixmap)
        layout.addWidget(icon_label)
        text_label = QLabel(text, container)
        layout.addWidget(text_label)
        return container

    def _configure_tool_button(self, button: QToolButton) -> None:
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setIconSize(QSize(18, 18))
        button.setMinimumHeight(28)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.setAutoRaise(False)

    def _ensure_disabled_icon(self, icon: QIcon) -> None:
        probe = icon.pixmap(18, 18, QIcon.Mode.Disabled, QIcon.State.Off)
        if not probe.isNull():
            return
        normal = icon.pixmap(18, 18, QIcon.Mode.Normal, QIcon.State.Off)
        if normal.isNull():
            return
        disabled = QPixmap(normal.size())
        disabled.fill(Qt.GlobalColor.transparent)
        painter = QPainter(disabled)
        painter.setOpacity(0.35)
        painter.drawPixmap(0, 0, normal)
        painter.end()
        icon.addPixmap(disabled, QIcon.Mode.Disabled, QIcon.State.Off)

    def _load_icon(self, name: str) -> Optional[QIcon]:
        if name in self._icon_cache:
            return self._icon_cache[name]
        path = self._icon_dir / f"{name}.svg"
        if not path.exists():
            return None
        icon = QIcon(str(path))
        self._icon_cache[name] = icon
        return icon

    def _load_pixmap(self, name: str, size: int) -> Optional[QPixmap]:
        cache_key = f"{name}:{size}"
        if cache_key in self._pixmap_cache:
            return self._pixmap_cache[cache_key]
        path = self._icon_dir / f"{name}.svg"
        if not path.exists():
            return None
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return None
        pixmap = pixmap.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._pixmap_cache[cache_key] = pixmap
        return pixmap

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _load_assets(self) -> None:
        """Load grayscale image + mask and render them."""
        try:
            image = self._read_grayscale(self.image_path)
            mask = self._read_grayscale(self.mask_path)
        except ValueError as exc:
            logger.exception("Failed to load mask editor assets: %s", exc)
            self._set_status(str(exc))
            self.button_box.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
            return

        if image.shape != mask.shape:
            message = (
                f"Image/mask dimensions differ: {image.shape} vs {mask.shape}. "
                "Cannot edit mask until they match."
            )
            logger.error(message)
            self._set_status(message)
            self.button_box.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
            return

        self._image_array = image
        self._mask_array = (mask > 0).astype(np.uint8) * 255
        self._clear_history()
        self._mask_dirty = False
        self._image_dirty = False
        self._set_status(
            f"Loaded {image.shape[1]}×{image.shape[0]} image. Use brush/eraser to edit mask."
        )
        self._update_canvas()

    def _update_canvas(self) -> None:
        if self._image_array is None or self._mask_array is None:
            return

        self.image_item.setImage(self._image_array, autoLevels=True)
        self.mask_item.setImage(self._mask_array, autoLevels=False)
        self.view_box.autoRange()
        self._update_cursor_visual()
        if self._active_tool == "crop":
            self._init_crop_roi()

    @staticmethod
    def _read_grayscale(path: Path) -> np.ndarray:
        array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise ValueError(f"Failed to load grayscale image: {path}")
        if array.ndim != 2:
            raise ValueError(f"Expected single-channel image at {path}, got shape {array.shape}")
        return array.astype(np.uint8)

    def _set_active_tool(self, tool: str) -> None:
        # Leaving crop mode hides ROI if switching away
        if self._active_tool == "crop" and tool != "crop":
            self._exit_crop_mode()

        self._active_tool = tool
        if tool == "move":
            self._cursor_view_pos = None
            self.cursor_item.setVisible(False)
            self.canvas_viewport.setCursor(Qt.CursorShape.OpenHandCursor)
        elif tool == "crop":
            self.canvas_viewport.setCursor(Qt.CursorShape.ArrowCursor)
            self._enter_crop_mode()
        else:
            self.canvas_viewport.setCursor(Qt.CursorShape.BlankCursor)
        self._update_history_buttons()

    def _on_brush_size_changed(self, value: int) -> None:
        self._brush_radius = value
        if self.size_spin.value() != value:
            self.size_spin.blockSignals(True)
            self.size_spin.setValue(value)
            self.size_spin.blockSignals(False)
        self._update_cursor_visual()

    def _on_opacity_changed(self, value: int) -> None:
        self._overlay_opacity = max(0.0, min(1.0, value / 100.0))
        self.mask_item.setOpacity(self._overlay_opacity)
        if hasattr(self, "opacity_value_label"):
            self.opacity_value_label.setText(f"{value}%")

    # -------------------------------------------------------------- Crop
    def _init_crop_roi(self) -> None:
        """Create or reset the crop ROI to a sensible default."""
        if self._image_array is None:
            return
        height, width = self._image_array.shape
        if width <= 0 or height <= 0:
            return

        if self._crop_roi is None:
            default_w = max(10, int(width * 0.8))
            default_h = max(10, int(height * 0.8))
            start_x = (width - default_w) // 2
            start_y = (height - default_h) // 2
            self._crop_roi = pg.RectROI(
                [start_x, start_y],
                [default_w, default_h],
                maxBounds=QRectF(0, 0, width, height),
                movable=True,
                resizable=True,
                removable=False,
                pen=pg.mkPen(color=(200, 200, 200), width=1.5),
            )
            self._crop_roi.addScaleHandle([0, 0], [1, 1])
            self._crop_roi.addScaleHandle([1, 1], [0, 0])
            self._crop_roi.addScaleHandle([1, 0], [0, 1])
            self._crop_roi.addScaleHandle([0, 1], [1, 0])
            self._crop_roi.sigRegionChanged.connect(self._update_crop_status)
            self._crop_roi.setZValue(20)
            self.canvas_plot.addItem(self._crop_roi)

        # Reset bounds to current image size
        self._crop_roi.setSize([max(10, int(width * 0.8)), max(10, int(height * 0.8))], update=True)
        self._crop_roi.setPos((width - self._crop_roi.size()[0]) / 2, (height - self._crop_roi.size()[1]) / 2)
        self._crop_roi.maxBounds = QRectF(0, 0, width, height)
        self._crop_roi.setVisible(True)
        self._update_crop_status()

    def _enter_crop_mode(self) -> None:
        if self._image_array is None or self._mask_array is None:
            self.crop_button.blockSignals(True)
            self.crop_button.setChecked(False)
            self.crop_button.blockSignals(False)
            self._set_status("Load an image before cropping.")
            return
        self._init_crop_roi()
        self._set_status("Adjust crop, press Enter to apply, Esc to cancel.")

    def _exit_crop_mode(self) -> None:
        if self._crop_roi is not None:
            self._crop_roi.setVisible(False)
        if hasattr(self, "crop_button"):
            self.crop_button.blockSignals(True)
            self.crop_button.setChecked(False)
            self.crop_button.blockSignals(False)
        self._pending_crop_rect = None

    def _update_crop_status(self) -> None:
        if self._crop_roi is None or self._image_array is None:
            return
        pos = self._crop_roi.pos()
        size = self._crop_roi.size()
        x = max(0, int(round(pos.x())))
        y = max(0, int(round(pos.y())))
        w = max(1, int(round(size.x())))
        h = max(1, int(round(size.y())))
        height, width = self._image_array.shape
        # Clamp within bounds
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y
        self._pending_crop_rect = (x, y, w, h)
        self._set_status(f"Crop: ({x}, {y}) size {w}×{h}  | Image: {width}×{height}")

    def _apply_crop(self) -> None:
        if self._pending_crop_rect is None or self._image_array is None or self._mask_array is None:
            return
        x, y, w, h = self._pending_crop_rect
        if w <= 0 or h <= 0:
            return
        prev_image = self._image_array
        prev_mask = self._mask_array
        try:
            self._image_array = prev_image[y : y + h, x : x + w].copy()
            self._mask_array = prev_mask[y : y + h, x : x + w].copy()
        except Exception as exc:
            logger.exception("Failed to crop", exc_info=exc)
            QMessageBox.critical(self, "Crop", f"Crop failed: {exc}")
            return
        self._mask_dirty = True
        self._image_dirty = True
        self._clear_history()
        self._update_canvas()
        self._set_status(f"Cropped to {w}×{h} at ({x}, {y})")
        self._exit_crop_mode()
        if hasattr(self, "eraser_button"):
            self.eraser_button.setChecked(True)

    def _clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        self.undo_button.setEnabled(bool(self._undo_stack))
        self.redo_button.setEnabled(bool(self._redo_stack))

    def _confirm_discard(self) -> bool:
        if not (self._mask_dirty or self._image_dirty):
            return True
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Discard edits?")
        message.setText("You have unsaved edits. Discard changes?")
        message.setStandardButtons(
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
        )
        message.setDefaultButton(QMessageBox.StandardButton.Cancel)
        result = message.exec()
        return result == QMessageBox.StandardButton.Discard

    def _save_mask(self) -> bool:
        if self._mask_array is None:
            return True
        mask_to_save = self._mask_array.astype(np.uint8)
        success = cv2.imwrite(str(self.mask_path), mask_to_save)
        if not success:
            logger.error("Failed to save mask to %s", self.mask_path)
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not write mask to {self.mask_path}",
            )
            return False
        self.mask_saved.emit(self.mask_path)
        return True

    def _save_image(self) -> bool:
        if self._image_array is None:
            return True
        image_to_save = self._image_array.astype(np.uint8)
        success = cv2.imwrite(str(self.image_path), image_to_save)
        if not success:
            logger.error("Failed to save image to %s", self.image_path)
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not write image to {self.image_path}",
            )
            return False
        return True

    def accept(self) -> None:  # noqa: D401
        """Persist edits and close the dialog."""
        if self._mask_array is None:
            super().accept()
            return
        if not (self._mask_dirty or self._image_dirty):
            super().accept()
            return
        if self._save_mask() and self._save_image():
            self._mask_dirty = False
            self._image_dirty = False
            self._clear_history()
            self._set_status(f"Saved mask to {self.mask_path}")
            super().accept()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ensure dialog closes cleanly when user hits the titlebar close."""
        if self.result() in (QDialog.Accepted, QDialog.Rejected):
            super().closeEvent(event)
            return
        if self._confirm_discard():
            self._mask_dirty = False
            self._image_dirty = False
            self._clear_history()
            event.accept()
            super().reject()
        else:
            event.ignore()

    # -------------------------------------------------------------- Painting
    def eventFilter(self, obj, event):  # noqa: D401
        """Intercept viewport mouse events for painting + cursor preview."""
        if obj is self.canvas_viewport:
            if self._active_tool == "move":
                if event.type() == QEvent.Type.Enter:
                    self.canvas_viewport.setCursor(Qt.CursorShape.OpenHandCursor)
                if event.type() == QEvent.Type.MouseMove:
                    self._cursor_view_pos = None
                    self.cursor_item.setVisible(False)
                if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
                    if event.button() == Qt.MouseButton.LeftButton:
                        self.canvas_viewport.setCursor(Qt.CursorShape.ClosedHandCursor)
                if event.type() == QEvent.Type.MouseButtonRelease and isinstance(event, QMouseEvent):
                    self.canvas_viewport.setCursor(Qt.CursorShape.OpenHandCursor)
                if event.type() == QEvent.Type.Leave:
                    self.canvas_viewport.unsetCursor()
                return False
            elif self._active_tool == "crop":
                if event.type() == QEvent.Type.Enter:
                    self.canvas_viewport.setCursor(Qt.CursorShape.ArrowCursor)
                if event.type() == QEvent.Type.Leave:
                    self.canvas_viewport.unsetCursor()
                if event.type() == QEvent.Type.MouseMove:
                    return False
                if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
                    return False
            else:
                if event.type() == QEvent.Type.Enter:
                    self.canvas_viewport.setCursor(Qt.CursorShape.BlankCursor)
                if event.type() == QEvent.Type.Leave:
                    self.canvas_viewport.unsetCursor()
            if event.type() == QEvent.Type.MouseMove:
                return self._handle_mouse_move(event)
            if event.type() == QEvent.Type.MouseButtonPress:
                return self._handle_mouse_press(event)
            if event.type() == QEvent.Type.MouseButtonRelease:
                return self._handle_mouse_release(event)
            if event.type() == QEvent.Type.Leave:
                self._cursor_view_pos = None
                self.cursor_item.setVisible(False)
        return super().eventFilter(obj, event)

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._image_array is None
            or self._active_tool in ("move", "crop")
        ):
            return False
        mapping = self._map_event_positions(event)
        if mapping is None:
            return False
        paint_pos, view_pos = mapping
        self._cursor_view_pos = view_pos
        self._update_cursor_visual()
        self._begin_stroke_snapshot()
        self._is_painting = True
        self._last_paint_pos = paint_pos
        changed = self._apply_brush(*paint_pos)
        if changed:
            self._refresh_mask_item()
        event.accept()
        return True

    def _handle_mouse_move(self, event: QMouseEvent) -> bool:
        mapping = self._map_event_positions(event)
        if mapping is None:
            self._cursor_view_pos = None
            self._update_cursor_visual()
            if self._active_tool != "move":
                self.canvas_viewport.setCursor(Qt.CursorShape.ArrowCursor)
            return False

        paint_pos, view_pos = mapping
        if self._active_tool != "move":
            self.canvas_viewport.setCursor(Qt.CursorShape.BlankCursor)
        self._cursor_view_pos = view_pos
        self._update_cursor_visual()
        if not self._is_painting:
            return False
        changed = self._stroke_to(paint_pos)
        if changed:
            self._refresh_mask_item()
        event.accept()
        return True

    def _handle_mouse_release(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or self._active_tool == "move":
            return False
        if not self._is_painting:
            return False
        self._is_painting = False
        self._last_paint_pos = None
        if not self._stroke_changed and self._undo_stack:
            self._undo_stack.pop()
        self._update_history_buttons()
        event.accept()
        return True

    def _stroke_to(self, pos: tuple[float, float]) -> bool:
        if self._last_paint_pos is None:
            changed = self._apply_brush(*pos)
        else:
            changed = self._paint_line(self._last_paint_pos, pos)
        self._last_paint_pos = pos
        return changed

    def _begin_stroke_snapshot(self) -> None:
        if self._mask_array is None:
            return
        self._undo_stack.append(self._mask_array.copy())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._stroke_changed = False
        self._update_history_buttons()

    def _paint_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = (dx**2 + dy**2) ** 0.5
        step = max(self._brush_radius * 0.5, 1.0)
        samples = max(int(distance / step), 1)
        changed = False
        for i in range(samples + 1):
            t = i / samples
            x = start[0] + dx * t
            y = start[1] + dy * t
            if self._apply_brush(x, y):
                changed = True
        return changed

    def _apply_brush(self, x: float, y: float) -> bool:
        if self._mask_array is None:
            return False
        height, width = self._mask_array.shape
        cx = int(round(x))
        cy = int(round(y))
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            return False

        radius = self._brush_radius
        x_min = max(0, cx - radius)
        x_max = min(width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(height, cy + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            return False

        sub = self._mask_array[y_min:y_max, x_min:x_max]
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        value = 255 if self._active_tool == "brush" else 0

        if value == 255:
            changed = np.any(sub[mask] != 255)
            sub[mask] = 255
        else:
            changed = np.any(sub[mask] != 0)
            sub[mask] = 0

        if changed:
            self._mask_dirty = True
            self._stroke_changed = True
        return changed

    def _refresh_mask_item(self) -> None:
        if self._mask_array is not None:
            self.mask_item.setImage(self._mask_array, autoLevels=False)
            self._update_cursor_visual()

    def _undo_action(self) -> None:
        if not self._undo_stack or self._mask_array is None:
            return
        snapshot = self._undo_stack.pop()
        self._redo_stack.append(self._mask_array.copy())
        self._mask_array = snapshot
        self._mask_dirty = bool(self._undo_stack)
        self._refresh_mask_item()
        self._update_history_buttons()

    def _redo_action(self) -> None:
        if not self._redo_stack or self._mask_array is None:
            return
        snapshot = self._redo_stack.pop()
        self._undo_stack.append(self._mask_array.copy())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._mask_array = snapshot
        self._mask_dirty = True
        self._refresh_mask_item()
        self._update_history_buttons()

    def _map_event_positions(
        self, event: QMouseEvent
    ) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        """Return (paint_coords, view_coords) for the given mouse event."""
        if self._image_array is None:
            return None

        pos = event.position()
        scene_pos = self.canvas_widget.mapToScene(pos.toPoint())
        view_point = self.view_box.mapSceneToView(scene_pos)
        view_x = view_point.x()
        view_y = view_point.y()
        if np.isnan(view_x) or np.isnan(view_y):
            return None

        local_point = self.image_item.mapFromScene(scene_pos)
        paint_x = local_point.x()
        paint_y = local_point.y()
        if np.isnan(paint_x) or np.isnan(paint_y):
            return None

        height, width = self._image_array.shape
        if paint_x < 0 or paint_y < 0 or paint_x >= width or paint_y >= height:
            return None

        return (paint_x, paint_y), (view_x, view_y)

    def _update_cursor_visual(self) -> None:
        if (
            self._cursor_view_pos is None
            or self._image_array is None
            or self._active_tool == "move"
        ):
            self.cursor_item.setVisible(False)
            return
        radius = self._brush_radius
        x, y = self._cursor_view_pos
        self.cursor_item.setRect(
            x - radius,
            y - radius,
            radius * 2,
            radius * 2,
        )
        self.cursor_item.setVisible(True)

    def _handle_cancel(self) -> None:
        if not (self._mask_dirty or self._image_dirty) or self._confirm_discard():
            self._mask_dirty = False
            self._image_dirty = False
            self._clear_history()
            super().reject()

    # -------------------------------------------------------------- Keys
    def keyPressEvent(self, event) -> None:
        if self._active_tool == "crop":
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._apply_crop()
                return
            if event.key() == Qt.Key.Key_Escape:
                self._exit_crop_mode()
                if hasattr(self, "eraser_button"):
                    self.eraser_button.setChecked(True)
                return
        return super().keyPressEvent(event)
