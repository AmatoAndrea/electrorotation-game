"""Asset management controller for backgrounds and cell templates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QWidget,
)

from ...catalog import list_backgrounds, list_cell_lines, list_cell_templates
from ...settings import asset_root


class AssetController:
    """Manages asset loading, filtering, and thumbnail generation."""

    def __init__(self, parent: QWidget):
        self.parent = parent
        self.thumbnail_icons: Dict[Path, QIcon] = {}
        self.thumbnail_sizes: Dict[Path, Tuple[int, int]] = {}
        self._background_paths: List[Path] = []

    def reload_all_assets(
        self,
        background_list: QListWidget | None,
        cell_line_combo,
        search_text: str = "",
        on_line_changed=None,
    ) -> None:
        """Reload backgrounds and cell lines from asset directories."""
        # Load backgrounds
        if background_list is not None:
            try:
                self._background_paths = list_backgrounds()
            except FileNotFoundError as err:
                QMessageBox.critical(self.parent, "Assets", str(err))
                self._background_paths = []
            self.filter_backgrounds(background_list, search_text)

        # Load cell lines
        cell_line_combo.blockSignals(True)
        cell_line_combo.clear()
        try:
            lines = list_cell_lines()
        except FileNotFoundError as err:
            QMessageBox.critical(self.parent, "Assets", str(err))
            lines = []
        
        for line_dir in lines:
            cell_line_combo.addItem(line_dir.name)
        cell_line_combo.blockSignals(False)

    def filter_backgrounds(
        self,
        background_list: QListWidget | None,
        filter_text: str,
        selected_path: Optional[Path] = None,
    ) -> None:
        """Filter and display backgrounds matching search text."""
        if background_list is None:
            return
        filter_text = filter_text.strip().lower()
        background_list.clear()
        
        if not self._background_paths:
            return

        matched_item: Optional[QListWidgetItem] = None
        for path in self._background_paths:
            if filter_text and filter_text not in path.name.lower():
                continue
            
            display = path.relative_to(asset_root()).as_posix()
            item = QListWidgetItem(display)
            icon, _ = self.get_icon_with_size(path)
            if icon:
                item.setIcon(icon)
            item.setData(Qt.ItemDataRole.UserRole, path)
            background_list.addItem(item)
            
            if selected_path and path == selected_path:
                matched_item = item

        if matched_item:
            background_list.setCurrentItem(matched_item)
            background_list.scrollToItem(matched_item)

    def populate_templates_for_line(
        self,
        line_name: str,
        template_list: QListWidget,
    ) -> None:
        """Populate cell template list for given cell line."""
        template_list.clear()
        try:
            templates = list_cell_templates(line_name)
        except FileNotFoundError as err:
            QMessageBox.critical(self.parent, "Assets", str(err))
            return
        
        for path in templates:
            icon, size = self.get_icon_with_size(path)
            label = f"{path.name}\n{size[0]}x{size[1]}"
            item = QListWidgetItem(label)
            if icon:
                item.setIcon(icon)
            item.setData(Qt.ItemDataRole.UserRole, path)
            template_list.addItem(item)

    def get_icon_with_size(self, path: Path) -> Tuple[Optional[QIcon], Tuple[int, int]]:
        """Get or create cached thumbnail icon for asset path."""
        if path in self.thumbnail_icons:
            return self.thumbnail_icons[path], self.thumbnail_sizes[path]
        
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return None, (0, 0)
        
        icon = QIcon(pixmap.scaled(78, 78, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        size = (pixmap.width(), pixmap.height())
        self.thumbnail_icons[path] = icon
        self.thumbnail_sizes[path] = size
        return icon, size

    def get_path_from_item(self, item: QListWidgetItem) -> Optional[Path]:
        """Extract Path from list widget item data."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(path, Path):
            try:
                return Path(path)
            except (TypeError, ValueError):
                return None
        return path
