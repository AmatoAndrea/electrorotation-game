"""Custom item delegates for table editing."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QStyledItemDelegate, QWidget


class FloatItemDelegate(QStyledItemDelegate):
    """Custom delegate for editing floating-point values in table cells.
    
    Attributes:
        decimals: Number of decimal places to display
        minimum: Minimum allowed value
        maximum: Maximum allowed value
    """
    
    def __init__(
        self,
        decimals: int = 3,
        minimum: float = -1e6,
        maximum: float = 1e6,
        parent: Optional[QWidget] = None,
        suffix: Optional[str] = None,
    ):
        """Initialize the delegate.
        
        Args:
            decimals: Number of decimal places (default: 3)
            minimum: Minimum allowed value (default: -1e6)
            maximum: Maximum allowed value (default: 1e6)
            parent: Parent widget
        """
        super().__init__(parent)
        self._decimals = decimals
        self._min = minimum
        self._max = maximum
        self._suffix = suffix or ""

    def createEditor(self, parent, option, index):  # noqa: D401
        """Create a QDoubleSpinBox editor for the table cell."""
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(self._decimals)
        editor.setRange(self._min, self._max)
        editor.setSingleStep(0.1)
        if self._suffix:
            editor.setSuffix(f" {self._suffix}")
        editor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return editor

    def setEditorData(self, editor, index):  # noqa: D401
        """Load data from the model into the editor."""
        try:
            value = float(index.model().data(index, Qt.ItemDataRole.EditRole))
        except (TypeError, ValueError):
            value = 0.0
        editor.setValue(value)

    def setModelData(self, editor, model, index):  # noqa: D401
        """Save data from the editor back to the model."""
        value = float(editor.value())
        model.setData(index, value, Qt.ItemDataRole.EditRole)
        model.setData(index, value, Qt.ItemDataRole.DisplayRole)

    def displayText(self, value, locale):  # noqa: D401
        """Format display string with suffix."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return super().displayText(value, locale)
        text = f"{numeric:.3f}"
        if self._suffix:
            text = f"{text} {self._suffix}"
        return text
