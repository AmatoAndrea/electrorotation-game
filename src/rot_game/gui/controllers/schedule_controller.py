"""Schedule interval management controller."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import QMessageBox, QWidget

from ..models import ScheduleInterval, ScenarioState


class ScheduleController:
    """Manages schedule interval operations (add, remove, duplicate)."""

    def __init__(self, parent: QWidget, state: ScenarioState):
        self.parent = parent
        self._state = state

    @property
    def state(self) -> ScenarioState:
        """Return the current scenario state, following the parent if it changes."""
        if self.parent is not None and hasattr(self.parent, "state"):
            return self.parent.state
        return self._state

    @state.setter
    def state(self, new_state: ScenarioState) -> None:
        """Allow explicit state updates when no parent is tracking state."""
        self._state = new_state

    def add_interval_at_end(self) -> None:
        """Add new interval at the end of schedule."""
        self.insert_interval_after(len(self.state.schedule) - 1)

    def insert_interval_after(self, index: int) -> None:
        """Insert new interval after given index."""
        insert_at = max(0, index + 1)
        new_interval = ScheduleInterval(
            frequency_khz=60.0,
            duration_s=2.0,
            angular_velocity_rad_s=3.0,
        )
        self.state.schedule.insert(insert_at, new_interval)

    def duplicate_interval(self, index: int) -> bool:
        """Duplicate interval at index, return success status."""
        if index < 0 or index >= len(self.state.schedule):
            return False
        
        current = self.state.schedule[index]
        duplicate = ScheduleInterval(
            frequency_khz=current.frequency_khz,
            duration_s=current.duration_s,
            angular_velocity_rad_s=current.angular_velocity_rad_s,
            delay_after_s=current.delay_after_s,
        )
        self.state.schedule.insert(index + 1, duplicate)
        return True

    def delete_interval(self, index: int) -> bool:
        """Delete interval at index, return success status."""
        if index < 0 or index >= len(self.state.schedule):
            return False
        
        if len(self.state.schedule) == 1:
            QMessageBox.information(
                self.parent,
                "Schedule",
                "At least one interval is required.",
            )
            return False
        
        self.state.schedule.pop(index)
        return True

    def update_interval(
        self,
        index: int,
        frequency_khz: Optional[float] = None,
        duration_s: Optional[float] = None,
        angular_velocity_rad_s: Optional[float] = None,
        delay_after_s: Optional[float] = None,
    ) -> bool:
        """Update interval fields, return success status."""
        if index < 0 or index >= len(self.state.schedule):
            return False
        
        interval = self.state.schedule[index]
        if frequency_khz is not None:
            interval.frequency_khz = frequency_khz
        if duration_s is not None:
            interval.duration_s = duration_s
        if angular_velocity_rad_s is not None:
            interval.angular_velocity_rad_s = angular_velocity_rad_s
        if delay_after_s is not None:
            interval.delay_after_s = delay_after_s
        
        return True

    def get_interval(self, index: int) -> Optional[ScheduleInterval]:
        """Get interval at index, or None if invalid."""
        if index < 0 or index >= len(self.state.schedule):
            return None
        return self.state.schedule[index]

    def calculate_total_duration(self) -> float:
        """Calculate total schedule duration in seconds."""
        return sum(
            interval.duration_s + interval.delay_after_s
            for interval in self.state.schedule
        )
