"""Schedule timeline visualization widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from ..models import ScenarioState
from ..utils import frequency_brush

if TYPE_CHECKING:
    from ..app import SimulatorEditor


class ScheduleTimelineWidget(pg.PlotWidget):
    """Interactive timeline visualization of frequency schedule intervals."""
    
    # Signal emitted when user clicks or drags to seek: (frame_number, time_seconds)
    seek_requested = Signal(int, float)

    def __init__(self, editor: SimulatorEditor):
        super().__init__()
        self.editor = editor

        # Blend with surrounding UI instead of default black background
        self.setBackground(None)
        
        plot_item = self.getPlotItem()

        # Hide left axis
        plot_item.hideAxis("left")

        # Theme-aware colors
        fg_color = self._theme_color()

        # Configure bottom axis
        axis = plot_item.getAxis("bottom")
        axis.setLabel("")
        axis.setStyle(showValues=True)
        axis.setPen(pg.mkPen(color=fg_color, width=2))

        plot_item.setMouseEnabled(x=True, y=False)
        self._schedule_items: List = []  # Track all schedule-related items for cleanup
        self._axis_color = fg_color
        self._interaction_enabled = False
        self._scene_click_connected = False
        
        # Playhead components
        self._playhead: Optional[pg.InfiniteLine] = None
        self._current_frame = 0
        self._total_frames = 0
        self._fps = 30.0
        self._playhead_label: Optional[pg.TextItem] = None
        
        # Initialize playhead
        self._create_playhead()

    def update_from_state(self, state: ScenarioState) -> None:
        """Update timeline visualization from scenario state."""
        plot = self.getPlotItem()
        
        # Remove all previously added schedule items (regions, labels, etc.)
        for item in self._schedule_items:
            plot.removeItem(item)
        self._schedule_items.clear()

        time_cursor = 0.0
        fps = state.fps
        self._fps = fps
        total = sum(interval.duration_s + interval.delay_after_s for interval in state.schedule)
        max_time = max(total, 1.0)
        default_span = 10.0
        actual_span = max(max_time, default_span)
        plot.setLimits(xMin=0, xMax=actual_span)
        plot.setXRange(0, actual_span)
        plot.setYRange(-0.5, 0.5, padding=0)

        # Calculate total frames
        total_frames = int(max_time * fps)
        self._total_frames = total_frames
        
        # Recreate playhead after clearing items
        self._create_playhead()
        
        # Reset playhead to start
        self.set_playhead_position(0, 0.0)

        for idx, interval in enumerate(state.schedule):
            start = time_cursor
            end = start + interval.duration_s
            if interval.frequency_khz == 0.0 and interval.angular_velocity_rad_s == 0.0:
                idle = pg.LinearRegionItem((start, end), brush=(200, 200, 200, 80))
                idle.setMovable(False)
                idle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # No interaction
                # Disable all interaction with region lines
                for line in idle.lines:
                    line.setMovable(False)
                    line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                plot.addItem(idle)
                self._schedule_items.append(idle)
                label = pg.TextItem("Idle", color=self._axis_color, anchor=(0.5, 0.5))
                label.setPos((start + end) / 2.0, 0.0)
                plot.addItem(label)
                self._schedule_items.append(label)
            else:
                region = pg.LinearRegionItem((start, end), brush=frequency_brush(interval.frequency_khz))
                region.setMovable(False)
                region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # No interaction
                # Disable all interaction with region lines (prevents dragging edges)
                for line in region.lines:
                    line.setMovable(False)
                    line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                # Disconnect the signal - regions are display-only
                # region.sigRegionChangeFinished.connect(...) # REMOVED - no editing via timeline
                plot.addItem(region)
                self._schedule_items.append(region)
                label = pg.TextItem(f"{interval.frequency_khz:.0f} kHz", color=self._axis_color, anchor=(0.5, 0.5))
                label.setPos((start + end) / 2.0, 0.0)
                plot.addItem(label)
                self._schedule_items.append(label)
            time_cursor = end + interval.delay_after_s
    
    def _create_playhead(self) -> None:
        """Create the playhead indicator line and label."""
        plot = self.getPlotItem()
        
        # Create vertical line for playhead - thick solid red line, draggable
        self._playhead = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(255, 80, 80), width=3),  # Thick solid red line
            movable=True,  # Allow dragging for scrubbing (controlled via set_interaction_enabled)
            bounds=[0, None]  # Can't drag before 0
        )
        # Set high z-value to ensure playhead is always on top of intervals
        self._playhead.setZValue(1000)
        # Enable/disable dragging based on current interaction state
        self._playhead.setMovable(self._interaction_enabled)
        # Connect to signal for drag updates
        self._playhead.sigDragged.connect(self._on_playhead_dragged)
        plot.addItem(self._playhead)
        self._schedule_items.append(self._playhead)
        
        # Create red triangle marker at top of playhead
        # Use smaller, narrower triangle (0.08 instead of 0.15)
        # Triangle pointing down at y=0.5 (top of timeline)
        triangle_size = 0.12  # Reduced from 0.15 for narrower triangle
        triangle = pg.PlotCurveItem(
            x=[0, -triangle_size, triangle_size, 0],  # Triangle pointing down
            y=[0.5 - triangle_size, 0.5, 0.5, 0.5 - triangle_size],
            pen=None,
            brush=pg.mkBrush(255, 80, 80),  # Red fill matching playhead
            fillLevel=0
        )
        # Set high z-value to ensure triangle is always on top of intervals
        triangle.setZValue(1001)  # Higher than playhead line
        # Store triangle as an attribute so we can update its position
        self._playhead_triangle = triangle
        self._playhead_triangle_size = triangle_size  # Store size for updates
        plot.addItem(triangle)
        self._schedule_items.append(triangle)
        
        # Create label showing current frame and time
        self._playhead_label = pg.TextItem(
            "Frame 0 | 0.00s",
            color=self._axis_color,
            anchor=(0.5, 1.2)  # Centered, above the timeline
        )
        # Set high z-value for label too
        self._playhead_label.setZValue(1002)
        self._playhead_label.setPos(0, 0.5)
        plot.addItem(self._playhead_label)
        self._schedule_items.append(self._playhead_label)
        
        # Listen for scene clicks to allow seek on single clicks anywhere on the timeline
        if self.scene() is not None and not self._scene_click_connected:
            self.scene().sigMouseClicked.connect(self._on_scene_mouse_clicked)
            self._scene_click_connected = True
    
    def _on_playhead_dragged(self, line) -> None:
        """Handle playhead dragging (scrubbing).
        
        This is ONLY called when the user manually drags the playhead,
        not when it's updated programmatically during playback.
        
        Args:
            line: The InfiniteLine that was dragged
        """
        if not self._interaction_enabled:
            # Reset playhead back to start and ignore drag
            self.reset_to_start()
            return

        time_s = line.value()
        
        # Clamp to valid range
        if time_s < 0:
            time_s = 0.0
        
        # Convert to frame
        frame = int(time_s * self._fps)
        
        # Clamp frame to valid range
        if frame >= self._total_frames and self._total_frames > 0:
            frame = self._total_frames - 1
        if frame < 0:
            frame = 0
        
        # Recalculate actual time from frame (for snapping to frame boundaries)
        actual_time = frame / self._fps if self._fps > 0 else 0.0
        
        # Update triangle position (use stored triangle_size to maintain consistent size)
        if hasattr(self, '_playhead_triangle') and self._playhead_triangle:
            triangle_size = getattr(self, '_playhead_triangle_size', 0.08)
            self._playhead_triangle.setData(
                x=[actual_time, actual_time - triangle_size, actual_time + triangle_size, actual_time],
                y=[0.5 - triangle_size, 0.5, 0.5, 0.5 - triangle_size]
            )
        
        # Update label
        if self._playhead_label:
            self._playhead_label.setText(f"Frame {frame} | {actual_time:.2f}s")
            self._playhead_label.setPos(actual_time, 0.5)
        
        # Emit signal for seek - this will pause playback and seek
        self.seek_requested.emit(frame, actual_time)
    
    def set_playhead_position(self, frame: int, time_s: float) -> None:
        """Update playhead position to the given frame and time.
        
        Args:
            frame: Current frame number (0-based)
            time_s: Current time in seconds
        """
        self._current_frame = frame
        
        if self._playhead:
            # Update position - this won't trigger sigDragged (only user drags do)
            self._playhead.setValue(time_s)
        
        # Update triangle position (use stored triangle_size to maintain consistent size)
        if hasattr(self, '_playhead_triangle') and self._playhead_triangle:
            triangle_size = getattr(self, '_playhead_triangle_size', 0.08)
            self._playhead_triangle.setData(
                x=[time_s, time_s - triangle_size, time_s + triangle_size, time_s],
                y=[0.5 - triangle_size, 0.5, 0.5, 0.5 - triangle_size]
            )
        
        if self._playhead_label:
            self._playhead_label.setText(f"Frame {frame} | {time_s:.2f}s")
            self._playhead_label.setPos(time_s, 0.5)
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse clicks for seek-to-position."""
        if not self._interaction_enabled:
            event.ignore()
            return
        # Validate that we have all required data before allowing timeline interaction
        if not hasattr(self.editor, 'state'):
            super().mousePressEvent(event)
            return
        
        state = self.editor.state
        has_background = state.background is not None and state.background.exists()
        has_cells = len(state.cells) > 0
        has_schedule = len(state.schedule) > 0
        
        # Timeline is only interactive when all three elements are present
        if not (has_background and has_cells and has_schedule):
            # Ignore clicks - don't allow seeking
            super().mousePressEvent(event)
            return
        
        # Get click position in data coordinates
        plot = self.getPlotItem()
        vb = plot.vb
        mouse_point = vb.mapSceneToView(event.pos())
        clicked_time = mouse_point.x()
        
        # Clamp to valid range
        if clicked_time < 0:
            clicked_time = 0.0
        
        # Convert time to frame number
        frame = int(clicked_time * self._fps)
        
        # Clamp frame to valid range
        if frame >= self._total_frames and self._total_frames > 0:
            frame = self._total_frames - 1
        if frame < 0:
            frame = 0
        
        # Update playhead position
        actual_time = frame / self._fps if self._fps > 0 else 0.0
        self.set_playhead_position(frame, actual_time)
        
        # Emit signal for seek
        self.seek_requested.emit(frame, actual_time)
        
        # Call parent implementation
        super().mousePressEvent(event)

    def _on_scene_mouse_clicked(self, event) -> None:
        """Handle scene-level clicks to support seeking even when items eat the event."""
        if not self._interaction_enabled:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        # Map scene position to view coordinates
        plot = self.getPlotItem()
        vb = plot.vb
        mouse_point = vb.mapSceneToView(event.scenePos())
        time_s = max(0.0, mouse_point.x())
        frame = int(time_s * self._fps)
        if frame >= self._total_frames and self._total_frames > 0:
            frame = self._total_frames - 1
        actual_time = frame / self._fps if self._fps > 0 else 0.0
        self.set_playhead_position(frame, actual_time)
        self.seek_requested.emit(frame, actual_time)

    def set_interaction_enabled(self, enabled: bool) -> None:
        """Enable or disable user interaction with the timeline/playhead."""
        self._interaction_enabled = enabled
        if self._playhead:
            self._playhead.setMovable(enabled)
        if not enabled:
            # Ensure playhead rests at start when interaction is disabled
            self.reset_to_start(ensure_visible=True)

    def reset_to_start(self, ensure_visible: bool = True) -> None:
        """Reset playhead position to the start of the timeline."""
        time_s = 0.0
        self.set_playhead_position(0, time_s)
        if ensure_visible:
            self.show_playhead()
    
    def update_total_frames(self, total_frames: int, fps: float) -> None:
        """Update the total frame count and FPS for frame calculation.
        
        Args:
            total_frames: Total number of frames in the simulation
            fps: Frames per second
        """
        self._total_frames = total_frames
        self._fps = fps

    def hide_playhead(self) -> None:
        """Hide the playhead and reset to frame 0 when exiting preview mode."""
        if self._playhead:
            self._playhead.setVisible(False)
        if hasattr(self, '_playhead_triangle') and self._playhead_triangle:
            self._playhead_triangle.setVisible(False)
        if self._playhead_label:
            self._playhead_label.setVisible(False)
        self._current_frame = 0

    def show_playhead(self) -> None:
        """Show the playhead when entering preview mode."""
        if self._playhead:
            self._playhead.setVisible(True)
        if hasattr(self, '_playhead_triangle') and self._playhead_triangle:
            self._playhead_triangle.setVisible(True)
        if self._playhead_label:
            self._playhead_label.setVisible(True)

    def _theme_color(self) -> Tuple[int, int, int]:
        """Get theme-aware foreground color for text and axes."""
        palette = self.editor.palette() if hasattr(self.editor, "palette") else QApplication.instance().palette()
        window_color = palette.color(QPalette.ColorRole.Window)
        brightness = (window_color.red() * 0.299 + window_color.green() * 0.587 + window_color.blue() * 0.114)
        if brightness < 128:
            # Dark background - use light text
            return (240, 240, 240)
        # Light background - use dark text
        return (40, 40, 40)
