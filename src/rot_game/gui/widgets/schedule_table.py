"""Schedule table widget for editing frequency intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from .delegates import FloatItemDelegate


class ScheduleTableWidget(QWidget):
    """Table widget for editing simulation schedule intervals."""
    
    # Signals
    add_interval_clicked = Signal()
    remove_interval_clicked = Signal()
    load_schedule_clicked = Signal()
    context_menu_requested = Signal(object)  # QPoint

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._icon_dir = Path(__file__).resolve().parents[1] / "icons"
        self._icon_cache: dict[str, QIcon] = {}
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Buttons
        schedule_buttons = QHBoxLayout()
        self.load_schedule_button = QPushButton("Load Schedule")
        self._apply_icon(self.load_schedule_button, "table")
        self.add_interval_button = QPushButton("Add Interval")
        self._apply_icon(self.add_interval_button, "add")
        self.remove_interval_button = QPushButton("Remove Interval")
        self._apply_icon(self.remove_interval_button, "remove")
        schedule_buttons.addWidget(self.load_schedule_button)
        schedule_buttons.addWidget(self.add_interval_button)
        schedule_buttons.addWidget(self.remove_interval_button)
        layout.addLayout(schedule_buttons)

        # Schedule table
        self.schedule_table = QTableWidget()
        self.schedule_table.setColumnCount(5)
        self.schedule_table.setHorizontalHeaderLabels([
            "Frequency",
            "Duration",
            "Delay",
            "ω",
            "",
        ])
        self.schedule_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.schedule_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.schedule_table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.SelectedClicked
        )
        vertical_header = self.schedule_table.verticalHeader()
        vertical_header.setVisible(True)
        vertical_header.setDefaultSectionSize(24)
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.schedule_table.setAlternatingRowColors(True)
        self.schedule_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Set up delegates for float formatting with suffixes
        suffixes = {
            0: "kHz",
            1: "s",
            2: "s",
            3: "rad/s",
        }
        for column, suffix in suffixes.items():
            self.schedule_table.setItemDelegateForColumn(
                column,
                FloatItemDelegate(decimals=3, parent=self.schedule_table, suffix=suffix),
            )

        layout.addWidget(self.schedule_table)

        # Configure column stretching
        header = self.schedule_table.horizontalHeader()
        for col in range(4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

    def _connect_signals(self) -> None:
        self.load_schedule_button.clicked.connect(self.load_schedule_clicked.emit)
        self.add_interval_button.clicked.connect(self.add_interval_clicked.emit)
        self.remove_interval_button.clicked.connect(self.remove_interval_clicked.emit)
        self.schedule_table.customContextMenuRequested.connect(self.context_menu_requested.emit)

    def install_event_filter(self, filter_obj: object) -> None:
        """Install event filter on table and viewport for keyboard shortcuts."""
        self.schedule_table.installEventFilter(filter_obj)
        self.schedule_table.viewport().installEventFilter(filter_obj)

    def _apply_icon(self, button: QPushButton, name: str, size: int = 16) -> None:
        icon = self._load_icon(name)
        if icon:
            button.setIcon(icon)
            button.setIconSize(QSize(size, size))

    def _load_icon(self, name: str) -> Optional[QIcon]:
        if name in self._icon_cache:
            return self._icon_cache[name]
        path = self._icon_dir / f"{name}.svg"
        if not path.exists():
            return None
        icon = QIcon(str(path))
        self._icon_cache[name] = icon
        return icon
