"""Asset browser panel widget for backgrounds and cell templates."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Signal, Qt, QEvent
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFocusEvent,
    QIcon,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QStandardItem,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AssetBrowserWidget(QWidget):
    """Stacked background and cell template picker for the sidebar."""
    
    # Signals
    background_folder_clicked = Signal()
    cell_line_changed = Signal(str)
    template_double_clicked = Signal(object)  # QListWidgetItem
    template_context_menu_requested = Signal(object)
    add_template_requested = Signal()

    placeholder_text = "Select a cell line…"
    _add_tile_marker = "__cell_template_add_tile__"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._add_tile_icon: Optional[QIcon] = None

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        background_group = QGroupBox()
        background_layout = QVBoxLayout(background_group)
        background_layout.setContentsMargins(8, 8, 8, 8)
        background_layout.setSpacing(6)

        folder_row = QHBoxLayout()
        folder_row.setContentsMargins(0, 0, 0, 0)
        folder_row.setSpacing(6)
        background_label = QLabel("Background Frames:")
        background_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        folder_row.addWidget(background_label)
        self.background_folder_button = QPushButton("Select folder…")
        folder_row.addWidget(self.background_folder_button)
        folder_row.addStretch(1)
        background_layout.addLayout(folder_row)

        # Thumbnail + summary
        self.background_thumbnail = QLabel()
        self.background_thumbnail.setFixedSize(140, 90)
        self.background_thumbnail.setStyleSheet("background: #222; border: 1px solid #444;")
        self.background_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.background_summary = QLabel("No folder selected")
        self.background_summary.setWordWrap(True)
        self.background_summary.setStyleSheet("color: #ccc; font-size: 10pt;")

        thumb_row = QHBoxLayout()
        thumb_row.setContentsMargins(0, 0, 0, 0)
        thumb_row.setSpacing(8)
        thumb_row.addWidget(self.background_thumbnail)
        thumb_row.addWidget(self.background_summary, 1)
        background_layout.addLayout(thumb_row)

        self.background_warning = QLabel()
        self.background_warning.setWordWrap(True)
        self.background_warning.setStyleSheet("color: #FFA500; font-size: 10pt; padding-left: 4px;")
        self.background_warning.hide()
        background_layout.addWidget(self.background_warning)

        layout.addWidget(background_group)

        cells_group = QGroupBox()
        cells_layout = QVBoxLayout(cells_group)
        cells_layout.setContentsMargins(8, 8, 8, 8)
        cells_layout.setSpacing(6)

        cell_line_row = QHBoxLayout()
        cell_line_row.setContentsMargins(0, 0, 0, 0)
        cell_line_row.setSpacing(6)

        cell_line_label = QLabel("Cell Templates")
        cell_line_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        cell_line_row.addWidget(cell_line_label)

        self.cell_line_combo = QComboBox()
        placeholder = QStandardItem(self.placeholder_text)
        placeholder.setEnabled(False)
        placeholder.setData(self._placeholder_color(), Qt.ItemDataRole.ForegroundRole)
        model = self.cell_line_combo.model()
        model.appendRow(placeholder)
        self._placeholder_index = model.rowCount() - 1
        self.cell_line_combo.setCurrentIndex(self._placeholder_index)
        cell_line_row.addWidget(self.cell_line_combo, 1)
        cells_layout.addLayout(cell_line_row)

        self.cell_template_list = QListWidget()
        self.cell_template_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.cell_template_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.cell_template_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.cell_template_list.setIconSize(QSize(55, 55))
        self.cell_template_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # Disable drag-and-drop so thumbnails stay anchored in the grid.
        self.cell_template_list.setDragEnabled(False)
        self.cell_template_list.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.cell_template_list.setDropIndicatorShown(False)

        template_panel = QWidget()
        template_panel.setObjectName("CellTemplatePanel")
        template_panel.setStyleSheet("QWidget#CellTemplatePanel { background-color: #1a1a1a; border: 1px solid #222; }")
        template_panel_layout = QVBoxLayout(template_panel)
        template_panel_layout.setContentsMargins(4, 4, 4, 4)
        template_panel_layout.setSpacing(0)
        template_panel_layout.addWidget(self.cell_template_list)
        cells_layout.addWidget(template_panel)
        
        layout.addWidget(cells_group)
        layout.addStretch(1)

        self.cell_template_list.installEventFilter(self)

    def _connect_signals(self) -> None:
        self.background_folder_button.clicked.connect(self.background_folder_clicked.emit)
        self.cell_line_combo.currentTextChanged.connect(self.cell_line_changed.emit)
        self.cell_template_list.itemDoubleClicked.connect(self.template_double_clicked.emit)
        self.cell_template_list.customContextMenuRequested.connect(self.template_context_menu_requested.emit)
        self.cell_template_list.itemClicked.connect(self._on_template_item_clicked)
        self.cell_template_list.itemActivated.connect(self._on_template_item_activated)
    
    def ensure_cell_line_placeholder(self, selected: Optional[str]) -> None:
        model = self.cell_line_combo.model()
        existing_index = self.cell_line_combo.findText(self.placeholder_text)
        if existing_index == -1:
            placeholder = QStandardItem(self.placeholder_text)
            placeholder.setEnabled(False)
            placeholder.setData(self._placeholder_color(), Qt.ItemDataRole.ForegroundRole)
            model.insertRow(0, placeholder)
            placeholder_index = 0
        else:
            placeholder_index = existing_index
            placeholder_item = model.item(existing_index)
            placeholder_item.setEnabled(False)
            placeholder_item.setData(self._placeholder_color(), Qt.ItemDataRole.ForegroundRole)
            if existing_index != 0:
                row = model.takeRow(existing_index)
                model.insertRow(0, row)
                placeholder_index = 0

        self.cell_line_combo.blockSignals(True)
        if selected:
            target = self.cell_line_combo.findText(selected)
            if target != -1:
                self.cell_line_combo.setCurrentIndex(target)
            else:
                self.cell_line_combo.setCurrentIndex(placeholder_index)
        else:
            self.cell_line_combo.setCurrentIndex(placeholder_index)
        self.cell_line_combo.blockSignals(False)

    def eventFilter(self, obj, event):
        if obj is self.cell_template_list and event.type() == QEvent.Type.KeyPress:
            key = getattr(event, "key", lambda: None)()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                current = self.cell_template_list.currentItem()
                if self.is_add_template_item(current):
                    self.add_template_requested.emit()
                    return True
        if event.type() == QEvent.Type.FocusOut and obj in (self.cell_template_list,):
            if isinstance(event, QFocusEvent) and event.reason() == Qt.FocusReason.PopupFocusReason:
                return super().eventFilter(obj, event)
            obj.clearSelection()
        return super().eventFilter(obj, event)

    def set_background_thumbnail(self, pixmap: QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            self.background_thumbnail.setPixmap(QPixmap())
            self.background_thumbnail.setText("No preview")
        else:
            scaled = pixmap.scaled(self.background_thumbnail.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.background_thumbnail.setPixmap(scaled)
            self.background_thumbnail.setText("")

    def set_background_summary(self, text: str) -> None:
        self.background_summary.setText(text)

    def _placeholder_color(self):
        palette = self.palette()
        color = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        return QBrush(color)

    def ensure_add_template_tile(self) -> None:
        """Append the add-template tile to the end of the list (deduplicated)."""
        list_widget = self.cell_template_list
        for idx in reversed(range(list_widget.count())):
            if self.is_add_template_item(list_widget.item(idx)):
                list_widget.takeItem(idx)
        list_widget.addItem(self._build_add_template_item())

    def is_add_template_item(self, item) -> bool:
        if item is None:
            return False
        return bool(item.data(Qt.ItemDataRole.UserRole) == self._add_tile_marker)

    def _build_add_template_item(self):
        item = QListWidgetItem("New cell...")
        item.setData(Qt.ItemDataRole.UserRole, self._add_tile_marker)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        icon = self._add_tile_icon or self._make_add_tile_icon()
        if icon:
            item.setIcon(icon)
        item.setToolTip("New cell...")
        return item

    def _make_add_tile_icon(self) -> Optional[QIcon]:
        size = 78
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dark background with subtle border
        painter.fillRect(pixmap.rect(), QColor("#161616"))
        border_pen = QPen(QColor("#2a2a2a"))
        border_pen.setWidth(2)
        painter.setPen(border_pen)
        painter.drawRect(pixmap.rect().adjusted(1, 1, -2, -2))

        # Centered plus icon from add.svg
        icon_path = Path(__file__).resolve().parent.parent / "icons" / "add.svg"
        icon_size = int(size * 0.6)
        # Create tinted version of the icon
        glyph = QIcon(str(icon_path)).pixmap(icon_size, icon_size)
        tinted = QPixmap(icon_size, icon_size)
        tinted.fill(Qt.GlobalColor.transparent)
        tint_painter = QPainter(tinted)
        tint_painter.drawPixmap(0, 0, glyph)
        tint_painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        tint_painter.fillRect(tinted.rect(), QColor("#cfcfcf"))
        tint_painter.end()

        # Draw centered
        x = (size - icon_size) // 2
        y = (size - icon_size) // 2
        painter.drawPixmap(x, y, tinted)

        painter.end()
        self._add_tile_icon = QIcon(pixmap)
        return self._add_tile_icon

    def _on_template_item_clicked(self, item) -> None:
        if self.is_add_template_item(item):
            self.add_template_requested.emit()

    def _on_template_item_activated(self, item) -> None:
        if self.is_add_template_item(item):
            self.add_template_requested.emit()
