"""Configuration file I/O controller."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from ...config import SimulationConfig
from ... import load_simulation_config
from ...settings import asset_root
from ..models import ScenarioCell, ScenarioState, ScheduleInterval


class ConfigController:
    """Manages loading and saving simulation configurations."""

    def __init__(self, parent: QWidget):
        self.parent = parent

    def open_file_dialog(self) -> Optional[Path]:
        """Show file dialog for opening config, return path or None."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Open Simulation Config",
            str(asset_root()),
            "YAML Files (*.yaml *.yml)",
        )
        return Path(file_path) if file_path else None

    def load_from_file(self, path: Path) -> Optional[ScenarioState]:
        """Load configuration file and convert to ScenarioState."""
        try:
            config = load_simulation_config(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self.parent,
                "Open",
                f"Failed to load config: {exc}",
            )
            return None

        return self._config_to_state(config)

    def _config_to_state(self, config: SimulationConfig) -> ScenarioState:
        """Convert SimulationConfig to ScenarioState."""
        return ScenarioState(
            background=config.video.background,
            cells=[
                ScenarioCell(
                    id=cell.id,
                    template=cell.template,
                    mask=cell.mask.path,
                    trajectory_type=cell.trajectory.type,
                    start=tuple(cell.trajectory.start),
                    end=tuple(cell.trajectory.end),
                    control_points=list(cell.trajectory.control_points or []),
                )
                for cell in config.cells
            ],
            schedule=[
                ScheduleInterval(
                    frequency_khz=interval.frequency_khz,
                    duration_s=interval.duration_s,
                    angular_velocity_rad_s=interval.angular_velocity_rad_s or 0.0,
                    delay_after_s=interval.delay_after_s or 0.0,
                )
                for interval in config.schedule.intervals
            ],
            resolution=tuple(config.video.resolution),
            fps=config.video.fps,
            magnification=config.video.magnification,
            pixel_size_um=config.video.pixel_size_um,
            video_duration_s=config.video.duration_s,
        )

    def save_to_file(self, state: ScenarioState, path: Optional[Path] = None) -> Optional[Path]:
        """Save ScenarioState to file, return path or None on failure."""
        if path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Save Simulation Config",
                str(asset_root()),
                "YAML Files (*.yaml *.yml)",
            )
            if not file_path:
                return None
            path = Path(file_path)

        try:
            config, _ = state.to_config()
            # TODO: Implement config serialization
            # For now, this is a placeholder
            QMessageBox.information(
                self.parent,
                "Save",
                "Save functionality not yet implemented in refactored code.",
            )
            return path
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self.parent,
                "Save",
                f"Failed to save config: {exc}",
            )
            return None
