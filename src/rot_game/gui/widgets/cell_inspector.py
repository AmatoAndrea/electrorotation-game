"""Cell inspector panel widget for cell properties and trajectory editing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QKeySequence, QShortcut, QPalette, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Import trajectory registry to get available types
from rot_game.core.trajectory import get_registered_trajectory_types


class CellInspectorWidget(QWidget):
    """Inspector panel for viewing and editing cell properties."""
    
    # Signals
    cell_selected = Signal()
    remove_cells_clicked = Signal()
    scenario_context_menu_requested = Signal(object)
    edit_mask_requested = Signal()
    edit_mask_requested = Signal()
    rename_cell_requested = Signal(int)
    trajectory_type_changed = Signal(str)
    coordinate_changed = Signal(str, int, object)  # attribute, axis, spin_box
    control_point_changed = Signal(int, int, object)  # index, axis, spin_box
    trajectory_param_changed = Signal(str, object)  # param_name, value

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._icon_dir = Path(__file__).resolve().parents[1] / "icons"
        self._icon_cache: Dict[str, QIcon] = {}
        self._trajectory_options: List[Tuple[str, str, str]] = []
        self._dynamic_param_widgets: Dict[str, List[QWidget]] = {}  # param_name -> [label, widget(s)]
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        self._row_indices: Dict[str, int] = {}
        self._row_min_heights: Dict[str, int] = {}
        self._row_widgets: Dict[str, List[QWidget]] = {}

        # Scenario cells list + preview
        scenario_label = QLabel("Scenario Cells")
        layout.addWidget(scenario_label)

        list_panel = QWidget()
        list_panel.setObjectName("ScenarioListPanel")
        main_layout = QHBoxLayout(list_panel)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(2)

        preview_container = QWidget()
        preview_container.setObjectName("ScenarioPreviewContainer")
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(2, 2, 2, 2)
        preview_layout.setSpacing(2)

        self.cell_preview_label = QLabel()
        self.cell_preview_label.setFixedSize(80, 80)
        self.cell_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cell_preview_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        preview_layout.addWidget(self.cell_preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.edit_mask_button = QToolButton()
        self.edit_mask_button.setText("Edit Mask…")
        self.edit_mask_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.edit_mask_button.setFixedWidth(self.cell_preview_label.width())
        self._apply_button_icon(self.edit_mask_button, "edit")
        preview_layout.addWidget(self.edit_mask_button, alignment=Qt.AlignmentFlag.AlignCenter)

        preview_container.setStyleSheet(
            "QWidget#ScenarioPreviewContainer { background-color: transparent; border: none; }"
        )
        self.preview_container = preview_container
        self.preview_container.setVisible(False)
        main_layout.addWidget(preview_container, alignment=Qt.AlignmentFlag.AlignTop)

        self.scenario_cell_list = QListWidget()
        self.scenario_cell_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.scenario_cell_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        main_layout.addWidget(self.scenario_cell_list, 1)

        list_panel.setStyleSheet(
            "QWidget#ScenarioListPanel { background-color: #1a1a1a; border: 1px solid #222; }"
        )
        layout.addWidget(list_panel, 1)

        # Cell properties group (no header)
        cell_group = QGroupBox("")
        cell_layout = QGridLayout(cell_group)
        cell_layout.setContentsMargins(6, 6, 6, 6)
        cell_layout.setVerticalSpacing(8)
        cell_layout.setHorizontalSpacing(6)
        self._cell_layout = cell_layout
        
        # Cell ID label with bottom spacing
        self.cell_id_label = QLabel("Cell: -")
        cell_layout.addWidget(self.cell_id_label, 0, 0, 1, 3)

        # Trajectory (now takes full row since radius is removed)
        trajectory_row = QWidget()
        trajectory_layout = QHBoxLayout(trajectory_row)
        trajectory_layout.setContentsMargins(0, 0, 0, 0)
        trajectory_layout.setSpacing(12)
        
        self.trajectory_label = QLabel("Trajectory")
        trajectory_layout.addWidget(self.trajectory_label)
        self.trajectory_combo = QComboBox()
        self.trajectory_combo.setIconSize(QSize(16, 16))
        self._populate_trajectory_combo()
        self.trajectory_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        trajectory_layout.addWidget(self.trajectory_combo, 1)
        
        self.trajectory_row = trajectory_row
        cell_layout.addWidget(trajectory_row, 1, 0, 1, 3)  # Span all 3 columns

        row = 2
        cell_layout.setColumnStretch(1, 1)
        cell_layout.setColumnStretch(2, 1)
        
        # Start coordinates (will change to "Location" for stationary)
        self.start_label = QLabel("Start")
        cell_layout.addWidget(self.start_label, row, 0)
        self._row_indices["start"] = row
        self.start_x_spin = QDoubleSpinBox()
        self.start_x_spin.setDecimals(2)
        self.start_x_spin.setRange(-1000, 2000)
        self.start_x_spin.setValue(200)
        self.start_x_spin.setPrefix("X: ")
        self.start_x_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.start_x_spin, row, 1)
        
        self.start_y_spin = QDoubleSpinBox()
        self.start_y_spin.setDecimals(2)
        self.start_y_spin.setRange(-1000, 2000)
        self.start_y_spin.setValue(350)
        self.start_y_spin.setPrefix("Y: ")
        self.start_y_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.start_y_spin, row, 2)
        row += 1

        # End coordinates (hidden for stationary)
        self.end_label = QLabel("End")
        cell_layout.addWidget(self.end_label, row, 0)
        self._row_indices["end"] = row
        self.end_x_spin = QDoubleSpinBox()
        self.end_x_spin.setDecimals(2)
        self.end_x_spin.setRange(-1000, 2000)
        self.end_x_spin.setValue(450)
        self.end_x_spin.setPrefix("X: ")
        self.end_x_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.end_x_spin, row, 1)
        
        self.end_y_spin = QDoubleSpinBox()
        self.end_y_spin.setDecimals(2)
        self.end_y_spin.setRange(-1000, 2000)
        self.end_y_spin.setValue(150)
        self.end_y_spin.setPrefix("Y: ")
        self.end_y_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.end_y_spin, row, 2)
        row += 1

        # Control point 1
        self.control1_label = QLabel("Control 1")
        cell_layout.addWidget(self.control1_label, row, 0)
        self._row_indices["control1"] = row
        self.control1_x_spin = QDoubleSpinBox()
        self.control1_x_spin.setDecimals(2)
        self.control1_x_spin.setRange(-1000, 2000)
        self.control1_x_spin.setPrefix("X: ")
        self.control1_x_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.control1_x_spin, row, 1)
        
        self.control1_y_spin = QDoubleSpinBox()
        self.control1_y_spin.setDecimals(2)
        self.control1_y_spin.setRange(-1000, 2000)
        self.control1_y_spin.setPrefix("Y: ")
        self.control1_y_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.control1_y_spin, row, 2)
        row += 1

        # Control point 2
        self.control2_label = QLabel("Control 2")
        cell_layout.addWidget(self.control2_label, row, 0)
        self._row_indices["control2"] = row
        self.control2_x_spin = QDoubleSpinBox()
        self.control2_x_spin.setDecimals(2)
        self.control2_x_spin.setRange(-1000, 2000)
        self.control2_x_spin.setPrefix("X: ")
        self.control2_x_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.control2_x_spin, row, 1)
        
        self.control2_y_spin = QDoubleSpinBox()
        self.control2_y_spin.setDecimals(2)
        self.control2_y_spin.setRange(-1000, 2000)
        self.control2_y_spin.setPrefix("Y: ")
        self.control2_y_spin.setMinimumWidth(50)
        cell_layout.addWidget(self.control2_y_spin, row, 2)

        layout.addWidget(cell_group)
        layout.addStretch(1)
        self._initialize_row_padding(cell_layout)
        self._shortcuts: List[QShortcut] = []

    def set_preview_visible(self, visible: bool) -> None:
        if hasattr(self, "preview_container"):
            self.preview_container.setVisible(visible)
        if not visible and hasattr(self, "cell_preview_label"):
            self.cell_preview_label.clear()

    def _connect_signals(self) -> None:
        self.scenario_cell_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        self.scenario_cell_list.customContextMenuRequested.connect(self.scenario_context_menu_requested.emit)
        self.trajectory_combo.currentIndexChanged.connect(self._emit_trajectory_changed)
        edit_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        edit_shortcut.activated.connect(self.edit_mask_requested.emit)
        self._shortcuts.append(edit_shortcut)
        if hasattr(self, "edit_mask_button"):
            self.edit_mask_button.clicked.connect(self.edit_mask_requested.emit)
        self.scenario_cell_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # Position signals
        self.start_x_spin.valueChanged.connect(
            lambda: self.coordinate_changed.emit("start", 0, self.start_x_spin)
        )
        self.start_y_spin.valueChanged.connect(
            lambda: self.coordinate_changed.emit("start", 1, self.start_y_spin)
        )
        self.end_x_spin.valueChanged.connect(
            lambda: self.coordinate_changed.emit("end", 0, self.end_x_spin)
        )
        self.end_y_spin.valueChanged.connect(
            lambda: self.coordinate_changed.emit("end", 1, self.end_y_spin)
        )
        
        # Control point signals
        self.control1_x_spin.valueChanged.connect(
            lambda: self.control_point_changed.emit(0, 0, self.control1_x_spin)
        )
        self.control1_y_spin.valueChanged.connect(
            lambda: self.control_point_changed.emit(0, 1, self.control1_y_spin)
        )
        self.control2_x_spin.valueChanged.connect(
            lambda: self.control_point_changed.emit(1, 0, self.control2_x_spin)
        )
        self.control2_y_spin.valueChanged.connect(
            lambda: self.control_point_changed.emit(1, 1, self.control2_y_spin)
        )

    def _emit_trajectory_changed(self, index: int) -> None:
        if index < 0:
            return
        value = self.trajectory_combo.itemData(index, Qt.ItemDataRole.UserRole)
        if value is None:
            value = self.trajectory_combo.itemText(index).lower()
        self.trajectory_type_changed.emit(str(value))

    def _on_list_selection_changed(self) -> None:
        self._update_item_selection_styles()
        self.cell_selected.emit()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        row = self.scenario_cell_list.row(item)
        if row >= 0:
            self.rename_cell_requested.emit(row)

    def _initialize_row_padding(self, layout: QGridLayout) -> None:
        def capture(row_key: str, widgets: List[QWidget]) -> None:
            row_index = self._row_indices.get(row_key)
            if row_index is None:
                return
            height = max(w.sizeHint().height() for w in widgets)
            self._row_min_heights[row_key] = height
            layout.setRowMinimumHeight(row_index, height)

        def register_row(key: str, widgets: List[QWidget]) -> None:
            capture(key, widgets)
            self._row_widgets[key] = widgets

        register_row("end", [self.end_label, self.end_x_spin, self.end_y_spin])
        register_row("control1", [self.control1_label, self.control1_x_spin, self.control1_y_spin])
        register_row("control2", [self.control2_label, self.control2_x_spin, self.control2_y_spin])

    def _set_row_visibility(self, row_key: str, visible: bool) -> None:
        widgets = self._row_widgets.get(row_key)
        if not widgets:
            return
        for widget in widgets:
            widget.setVisible(visible)
        idx = self._row_indices.get(row_key)
        if idx is not None:
            min_height = self._row_min_heights.get(row_key, 0)
            self._cell_layout.setRowMinimumHeight(idx, min_height if visible else 0)

    def _clear_dynamic_params(self) -> None:
        """Remove all dynamically created parameter widgets."""
        for widgets in self._dynamic_param_widgets.values():
            for widget in widgets:
                self._cell_layout.removeWidget(widget)
                widget.deleteLater()
        self._dynamic_param_widgets.clear()

    def _create_dynamic_params(
        self, 
        trajectory_type: str, 
        current_params: Dict[str, Any],
        cell_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create parameter widgets based on trajectory metadata.
        
        Parameters
        ----------
        trajectory_type : str
            The trajectory type name.
        current_params : dict
            Current parameter values for the cell.
        cell_data : dict, optional
            Additional cell data for custom widgets (start, end, magnification, etc.)
        """
        from rot_game.core.trajectory import get_trajectory_metadata
        
        # Clear existing dynamic params
        self._clear_dynamic_params()
        
        # Get metadata for the trajectory type
        meta = get_trajectory_metadata(trajectory_type)
        
        # Find the next available row after control2
        base_row = self._row_indices.get("control2", 5) + 1
        
        # If there's a custom widget factory, use it
        if meta.custom_widget_factory is not None and cell_data is not None:
            def update_callback(field_name: str, value: Any) -> None:
                self.trajectory_param_changed.emit(field_name, value)
            
            try:
                custom_widget = meta.custom_widget_factory(cell_data, update_callback)
                if custom_widget is not None:
                    self._cell_layout.addWidget(custom_widget, base_row, 0, 1, 3)
                    self._dynamic_param_widgets["__custom_widget__"] = [custom_widget]
                    base_row += 1
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to create custom widget: {e}")
        
        # Also create standard params_schema widgets
        if not meta.params_schema:
            return
        
        # Create widgets for each parameter
        for param_name, param_schema in meta.params_schema.items():
            param_type = param_schema.get("type", "float")
            label_text = param_schema.get("label", param_name)
            default_value = param_schema.get("default", 0)
            current_value = current_params.get(param_name, default_value)
            
            # Create label
            label = QLabel(label_text)
            self._cell_layout.addWidget(label, base_row, 0)
            
            widgets: List[QWidget] = [label]
            
            # Create appropriate widget based on type
            if param_type == "float":
                spin = QDoubleSpinBox()
                spin.setDecimals(2)
                spin.setRange(
                    param_schema.get("min", -1000000),
                    param_schema.get("max", 1000000)
                )
                spin.setValue(float(current_value))
                spin.valueChanged.connect(
                    lambda val, name=param_name: self.trajectory_param_changed.emit(name, val)
                )
                self._cell_layout.addWidget(spin, base_row, 1, 1, 2)
                widgets.append(spin)
            
            elif param_type == "int":
                spin = QSpinBox()
                spin.setRange(
                    int(param_schema.get("min", -1000000)),
                    int(param_schema.get("max", 1000000))
                )
                spin.setValue(int(current_value))
                spin.valueChanged.connect(
                    lambda val, name=param_name: self.trajectory_param_changed.emit(name, val)
                )
                self._cell_layout.addWidget(spin, base_row, 1, 1, 2)
                widgets.append(spin)
            
            elif param_type == "choice":
                combo = QComboBox()
                options = param_schema.get("options", [])
                combo.addItems(options)
                if current_value in options:
                    combo.setCurrentText(str(current_value))
                combo.currentTextChanged.connect(
                    lambda val, name=param_name: self.trajectory_param_changed.emit(name, val)
                )
                self._cell_layout.addWidget(combo, base_row, 1, 1, 2)
                widgets.append(combo)
            
            self._dynamic_param_widgets[param_name] = widgets
            base_row += 1

    def get_dynamic_param_values(self) -> Dict[str, Any]:
        """Get current values of all dynamic parameter widgets."""
        params = {}
        for param_name, widgets in self._dynamic_param_widgets.items():
            if param_name == "__note__":
                continue
            if len(widgets) < 2:
                continue
            widget = widgets[1]
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                params[param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[param_name] = widget.currentText()
        return params

    def decorate_scenario_item(self, item: QListWidgetItem) -> None:
        widget = QWidget()
        widget.setObjectName("ScenarioCellItem")
        widget.setAutoFillBackground(True)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel(item.text())
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setObjectName("ScenarioCellLabel")
        layout.addWidget(label)

        button = QToolButton()
        button.setText("")
        button.setToolTip("Remove cell")
        button.setAutoRaise(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setStyleSheet(
            "QToolButton { border: none; padding: 0 4px; border-radius: 3px; }"
            "QToolButton:hover { background-color: #d9534f; }"
        )
        self._apply_button_icon(button, "delete")
        layout.addWidget(button)

        button.clicked.connect(lambda: self._on_remove_button_clicked(item))
        item.setSizeHint(widget.sizeHint())
        self.scenario_cell_list.setItemWidget(item, widget)
        self._update_item_selection_styles()

    def _on_remove_button_clicked(self, item: QListWidgetItem) -> None:
        self.scenario_cell_list.setCurrentItem(item)
        self.remove_cells_clicked.emit()

    def _update_item_selection_styles(self) -> None:
        for index in range(self.scenario_cell_list.count()):
            item = self.scenario_cell_list.item(index)
            widget = self.scenario_cell_list.itemWidget(item)
            if not widget:
                continue
            palette = widget.palette()
            label = widget.findChild(QLabel, "ScenarioCellLabel")
            if label:
                if item.isSelected():
                    bg = self.palette().color(QPalette.ColorRole.Highlight).name()
                    fg = self.palette().color(QPalette.ColorRole.HighlightedText).name()
                else:
                    bg = "transparent"
                    fg = self.palette().color(QPalette.ColorRole.Text).name()
                label.setStyleSheet(
                    f"background-color: {bg};"
                    f"color: {fg};"
                    "padding: 2px; border-radius: 2px;"
                )

    def set_trajectory_value(self, trajectory: str) -> None:
        index = self._find_trajectory_index(trajectory)
        if index != -1:
            self.trajectory_combo.setCurrentIndex(index)

    def _find_trajectory_index(self, trajectory: str) -> int:
        trajectory = trajectory.lower()
        for idx in range(self.trajectory_combo.count()):
            data = self.trajectory_combo.itemData(idx, Qt.ItemDataRole.UserRole)
            text = self.trajectory_combo.itemText(idx).lower()
            if data == trajectory or text == trajectory:
                return idx
        return -1

    def _populate_trajectory_combo(self) -> None:
        """Populate trajectory combo box with built-in and registered custom types."""
        # Built-in trajectory types with icons
        builtin_options = [
            ("Stationary", "stationary", "point"),
            ("Linear", "linear", "diagonal_line"),
            ("Parabolic", "parabolic", "curve"),
            ("Cubic", "cubic", "route"),
        ]
        
        # Get custom registered types (from plugins)
        custom_types = get_registered_trajectory_types()
        
        # Build combined list
        self._trajectory_options = builtin_options.copy()
        
        # Add custom types with a generic icon
        for custom_type in custom_types:
            # Format name: "my_custom_type" -> "My Custom Type"
            label = custom_type.replace("_", " ").title()
            # Use a generic icon for all custom/plugin trajectory types
            icon_name = "extension"
            self._trajectory_options.append((label, custom_type, icon_name))
        
        self.trajectory_combo.clear()
        for label, value, icon_name in self._trajectory_options:
            icon = self._load_icon(icon_name)
            if icon:
                self.trajectory_combo.addItem(icon, label, userData=value)
            else:
                # No icon found, just add text
                self.trajectory_combo.addItem(label, userData=value)

    def _apply_button_icon(self, button: QToolButton, name: str, size: int = 16) -> None:
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
