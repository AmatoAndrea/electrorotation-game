"""Manages interactive ROI handles, markers, and guide lines for trajectory editing."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt

from ..models import ScenarioCell


class HandleManager:
    """Manages ROI handles, visual markers, and guide lines for trajectory editing.
    
    This class handles the creation and updating of:
    - Transparent ROI handles (for dragging)
    - Visible scatter plot markers (visual feedback)
    - Guide lines (showing Bézier curve control relationships)
    
    Attributes:
        plot_item: PyQtGraph plot item to add handles to
    """
    
    def __init__(self, plot_item: pg.PlotItem):
        """Initialize the handle manager.
        
        Args:
            plot_item: The PyQtGraph plot item to manage handles on
        """
        self.plot_item = plot_item
        self._handles: Dict[int, Dict[str, pg.ROI]] = {}
        self._handle_markers: Dict[int, Dict[str, pg.ScatterPlotItem]] = {}
        self._guide_lines: Dict[int, List[pg.PlotDataItem]] = {}
        self._active_handle: Optional[pg.ROI] = None  # Track currently interacting handle
    
    def clear_all(self) -> None:
        """Remove all handles, markers, and guide lines from the plot."""
        # Remove ROI handles
        for handle_map in self._handles.values():
            for roi in handle_map.values():
                self.plot_item.removeItem(roi)
        self._handles.clear()
        
        # Remove markers
        for marker_map in self._handle_markers.values():
            for marker in marker_map.values():
                self.plot_item.removeItem(marker)
        self._handle_markers.clear()
        
        # Remove guide lines
        for guide_list in self._guide_lines.values():
            for guide_line in guide_list:
                self.plot_item.removeItem(guide_line)
        self._guide_lines.clear()
    
    def create_handle(
        self,
        pos: Tuple[float, float],
        color: Tuple[int, int, int],
        callback: Callable[[Tuple[float, float]], None],
        size: int = 12,
    ) -> pg.ROI:
        """Create a transparent draggable ROI handle.
        
        Args:
            pos: (x, y) position tuple
            color: RGB color tuple (not used for transparent handle, kept for API compatibility)
            callback: Function called when handle position changes
            size: Size of the handle in pixels
            
        Returns:
            The created ROI handle
        """
        # Create transparent ROI centered at position
        roi = pg.ROI(
            [pos[0] - size/2, pos[1] - size/2],
            [size, size],
            pen=pg.mkPen((0, 0, 0, 0)),  # Fully transparent
            resizable=False,
            movable=True,
            removable=False,
        )
        roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        roi.setAcceptHoverEvents(False)  # Don't intercept hover events
        roi._handle_size = size
        
        def current_center(_roi) -> Tuple[float, float]:
            """Get center position of ROI."""
            top_left = roi.pos()
            center_x = top_left.x() + size / 2
            center_y = top_left.y() + size / 2
            return center_x, center_y
        
        def on_move(_roi=None):
            """Called during drag."""
            callback(current_center(roi))
        
        def on_region_change_started():
            """Track when handle interaction starts."""
            self._active_handle = roi
        
        def on_region_change_finished():
            """Track when handle interaction ends."""
            if self._active_handle == roi:
                self._active_handle = None
        
        roi.sigRegionChanged.connect(on_move)
        roi.sigRegionChangeStarted.connect(on_region_change_started)
        roi.sigRegionChangeFinished.connect(on_region_change_finished)
        self.plot_item.addItem(roi)
        return roi
    
    def is_any_handle_active(self) -> bool:
        """Check if any handle is currently being interacted with.
        
        Returns:
            True if a handle is being dragged or hovered, False otherwise
        """
        return self._active_handle is not None
    
    def create_marker(
        self,
        pos: Tuple[float, float],
        color: Tuple[int, int, int],
        symbol: str,
        size: int,
        brush: Optional[Tuple[int, int, int, int]] = None,
    ) -> pg.ScatterPlotItem:
        """Create a visible scatter plot marker.
        
        Args:
            pos: (x, y) position tuple
            color: RGB color tuple for outline
            symbol: Marker symbol ('o', 's', 'd', etc.)
            size: Marker size in pixels
            brush: RGBA brush color, or None for outline-only
            
        Returns:
            The created scatter plot marker
        """
        pen = pg.mkPen(color, width=2)
        if brush:
            brush_obj = pg.mkBrush(*brush)
        else:
            brush_obj = None
        
        marker = pg.ScatterPlotItem(
            [pos[0]], [pos[1]],
            symbol=symbol,
            size=size,
            pen=pen,
            brush=brush_obj,
        )
        self.plot_item.addItem(marker)
        return marker
    
    def update_marker_position(
        self,
        marker: pg.ScatterPlotItem,
        pos: Tuple[float, float],
    ) -> None:
        """Update the position of an existing marker.
        
        Args:
            marker: The scatter plot marker to update
            pos: New (x, y) position
        """
        marker.setData([pos[0]], [pos[1]])
    
    def create_guide_line(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
    ) -> pg.PlotDataItem:
        """Create a dashed guide line between two points.
        
        Args:
            start_pos: Starting (x, y) position
            end_pos: Ending (x, y) position
            
        Returns:
            The created guide line
        """
        pen = pg.mkPen((100, 255, 255, 80), width=2.5, style=Qt.PenStyle.DashLine, dash=[3, 3])
        guide_line = self.plot_item.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            pen=pen,
        )
        return guide_line
    
    def update_guide_line(
        self,
        guide_line: pg.PlotDataItem,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
    ) -> None:
        """Update the positions of an existing guide line.
        
        Args:
            guide_line: The guide line to update
            start_pos: New starting (x, y) position
            end_pos: New ending (x, y) position
        """
        guide_line.setData([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]])
    
    def update_guide_lines_for_cell(
        self,
        cell_idx: int,
        cell: ScenarioCell,
    ) -> None:
        """Update all guide lines for a cell based on current positions.
        
        Args:
            cell_idx: Index of the cell in the scenario
            cell: The cell with updated positions
        """
        if cell_idx not in self._guide_lines:
            return
        
        guide_lines = self._guide_lines[cell_idx]
        
        # Update guide line from start to first control point
        if len(cell.control_points) >= 1 and len(guide_lines) >= 1:
            self.update_guide_line(
                guide_lines[0],
                cell.start,
                cell.control_points[0],
            )
        
        # For parabolic: guide from control to end
        if cell.trajectory_type == "parabolic" and len(guide_lines) >= 2:
            self.update_guide_line(
                guide_lines[1],
                cell.control_points[0],
                cell.end,
            )
        
        # For cubic: guide from control1 to control2, and control2 to end
        if cell.trajectory_type == "cubic" and len(cell.control_points) >= 2:
            if len(guide_lines) >= 2:
                self.update_guide_line(
                    guide_lines[1],
                    cell.control_points[0],
                    cell.control_points[1],
                )
            if len(guide_lines) >= 3:
                self.update_guide_line(
                    guide_lines[2],
                    cell.control_points[1],
                    cell.end,
                )
    
    def get_handle(self, cell_idx: int, key: str) -> Optional[pg.ROI]:
        """Get a handle by cell index and key.
        
        Args:
            cell_idx: Index of the cell
            key: Handle key ('start', 'end', 'control1', 'control2')
            
        Returns:
            The ROI handle, or None if not found
        """
        return self._handles.get(cell_idx, {}).get(key)
    
    def set_handle(self, cell_idx: int, key: str, handle: pg.ROI) -> None:
        """Store a handle reference.
        
        Args:
            cell_idx: Index of the cell
            key: Handle key
            handle: The ROI handle to store
        """
        if cell_idx not in self._handles:
            self._handles[cell_idx] = {}
        self._handles[cell_idx][key] = handle
    
    def get_marker(self, cell_idx: int, key: str) -> Optional[pg.ScatterPlotItem]:
        """Get a marker by cell index and key.
        
        Args:
            cell_idx: Index of the cell
            key: Marker key
            
        Returns:
            The scatter plot marker, or None if not found
        """
        return self._handle_markers.get(cell_idx, {}).get(key)
    
    def set_marker(self, cell_idx: int, key: str, marker: pg.ScatterPlotItem) -> None:
        """Store a marker reference.
        
        Args:
            cell_idx: Index of the cell
            key: Marker key
            marker: The scatter plot marker to store
        """
        if cell_idx not in self._handle_markers:
            self._handle_markers[cell_idx] = {}
        self._handle_markers[cell_idx][key] = marker
    
    def add_guide_line(self, cell_idx: int, guide_line: pg.PlotDataItem) -> None:
        """Add a guide line to the collection for a cell.
        
        Args:
            cell_idx: Index of the cell
            guide_line: The guide line to add
        """
        if cell_idx not in self._guide_lines:
            self._guide_lines[cell_idx] = []
        self._guide_lines[cell_idx].append(guide_line)
    
    def get_guide_lines(self, cell_idx: int) -> List[pg.PlotDataItem]:
        """Get all guide lines for a cell.
        
        Args:
            cell_idx: Index of the cell
            
        Returns:
            List of guide lines for the cell
        """
        return self._guide_lines.get(cell_idx, [])
