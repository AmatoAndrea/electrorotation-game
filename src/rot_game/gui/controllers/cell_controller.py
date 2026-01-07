"""Cell management controller for adding, removing, and editing cells."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import QListWidget, QListWidgetItem, QMessageBox, QWidget

from ...catalog import find_mask_for_template
from ..models import ScenarioCell, ScenarioState


class CellController:
    """Manages cell CRUD operations and validation."""

    def __init__(self, parent: QWidget, state: ScenarioState):
        self.parent = parent
        self._state = state

    @property
    def state(self) -> ScenarioState:
        """Return the current scenario state, tracking the parent if available."""
        if self.parent is not None and hasattr(self.parent, "state"):
            # Access through parent to follow replacements (e.g., load_config)
            return self.parent.state
        return self._state

    @state.setter
    def state(self, new_state: ScenarioState) -> None:
        """Allow explicit state replacement when parent is unavailable."""
        self._state = new_state

    def add_cells_from_templates(
        self,
        template_paths: List[Path],
        refresh_callback=None,
    ) -> List[str]:
        """Add cells from template paths, return list of added IDs."""
        if not template_paths:
            QMessageBox.information(
                self.parent,
                "Add Cells",
                "Select one or more cell templates first.",
            )
            return []

        added_ids: List[str] = []
        for template in template_paths:
            mask = find_mask_for_template(template)
            if mask is None:
                QMessageBox.warning(
                    self.parent,
                    "Assets",
                    f"No mask found matching {template.name}",
                )
                continue
            
            base_id = template.stem
            unique_id = self._unique_cell_id(base_id)
            width, height = self.state.resolution
            resolution = (width, height)
            center = (float(width) / 2.0, float(height) / 2.0)

            cell = ScenarioCell(
                id=unique_id,
                template=template,
                mask=mask,
                trajectory_type="stationary",
                start=center,
                end=center,
            )
            cell.end = cell.default_end_for_resolution(resolution)
            cell.ensure_controls(resolution)
            self.state.cells.append(cell)
            added_ids.append(unique_id)

        if added_ids and refresh_callback:
            refresh_callback(added_ids[-1])

        return added_ids

    def remove_cells_by_indices(self, indices: List[int]) -> List[str]:
        """Remove cells at given indices, return list of removed IDs."""
        if not indices:
            return []
        
        # Sort in reverse to avoid index shifting
        removed = []
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.state.cells):
                cell = self.state.cells.pop(idx)
                removed.append(cell.id)
        
        return removed

    def get_cell_by_index(self, index: int) -> Optional[ScenarioCell]:
        """Get cell at index, or None if invalid."""
        if index < 0 or index >= len(self.state.cells):
            return None
        return self.state.cells[index]

    def validate_cell_bounds(
        self,
        cell: ScenarioCell,
        resolution: tuple[int, int],
    ) -> Optional[str]:
        """Check if cell trajectory fits within resolution, return warning message."""
        width, height = resolution
        
        for point_name, point in [
            ("start", cell.start),
            ("end", cell.end),
            *[(f"control{i+1}", cp) for i, cp in enumerate(cell.control_points)],
        ]:
            x, y = point
            if not (0 <= x <= width and 0 <= y <= height):
                return f"{cell.id} {point_name} point ({x:.1f}, {y:.1f}) is out of bounds"
        
        return None

    def _unique_cell_id(self, base: str) -> str:
        """Generate unique cell ID from base name."""
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

    def find_cell_line_for_template(self, template: Path) -> Optional[str]:
        """Extract cell line name from template path."""
        # Template structure: .../cell_lines/{line_name}/...
        try:
            parts = template.parts
            if "cell_lines" in parts:
                idx = parts.index("cell_lines")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        except (ValueError, IndexError):
            pass
        return None
