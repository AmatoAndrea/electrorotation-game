"""
Plugin system for rot-game exporters and trajectory generators.

This module provides a plugin architecture that allows external packages
to register custom exporters and trajectory types. Plugins are discovered
via Python entry points:
  - Exporters: "rot_game.exporters"
  - Trajectories: "rot_game.trajectories"

Example plugin registration in pyproject.toml:
    [project.entry-points."rot_game.exporters"]
    my_exporter = "my_package:MyExporterPlugin"
    
    [project.entry-points."rot_game.trajectories"]
    my_trajectory = "my_package:register_trajectories"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import SimulationConfig
    from ..core.renderer import GroundTruth
    from ..assets import CellTemplate

logger = logging.getLogger(__name__)


@dataclass
class CLIArgument:
    """Definition for a CLI argument that a plugin requires."""

    name: str
    """Argument name (e.g., '--my-option')"""

    type: type = str
    """Argument type (str, int, float, Path, bool)"""

    default: Any = None
    """Default value"""

    help: str = ""
    """Help text for the argument"""

    required: bool = False
    """Whether this argument is required"""

    action: Optional[str] = None
    """Argparse action (e.g., 'store_true')"""


@dataclass
class ExportResult:
    """Result returned by an exporter plugin."""

    output_path: Path
    """Path to the primary output (file or directory)"""

    message: str = ""
    """Optional message to display to the user"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata about the export"""


class ExporterPlugin(ABC):
    """
    Abstract base class for exporter plugins.

    Plugins must implement this interface to be discovered and used
    by the rot-game CLI.

    Example
    -------
    >>> class MyExporter(ExporterPlugin):
    ...     name = "my-exporter"
    ...     description = "Export data in my custom format"
    ...
    ...     def get_cli_arguments(self) -> List[CLIArgument]:
    ...         return [
    ...             CLIArgument("--my-option", type=str, default="value", help="My option"),
    ...         ]
    ...
    ...     def export(self, frames, ground_truth, config, output_dir, **kwargs) -> ExportResult:
    ...         # Do export...
    ...         return ExportResult(output_path=output_dir / "my_output")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this exporter (used in CLI: --export-<name>)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description shown in CLI help."""
        ...

    def get_cli_arguments(self) -> List[CLIArgument]:
        """
        Return list of CLI arguments this plugin requires.

        These arguments will be added to the 'render' command when
        the plugin is discovered.

        Returns
        -------
        List[CLIArgument]
            List of CLI argument definitions
        """
        return []

    @abstractmethod
    def export(
        self,
        frames: np.ndarray,
        ground_truth: "GroundTruth",
        config: "SimulationConfig",
        output_dir: Path,
        cell_templates: Sequence["CellTemplate"],
        rotated_masks: Optional[List[List[np.ndarray]]] = None,
        **kwargs: Any,
    ) -> ExportResult:
        """
        Export simulation data in the plugin's format.

        Parameters
        ----------
        frames : np.ndarray
            Rendered video frames (frames, height, width)
        ground_truth : GroundTruth
            Ground truth data containing cell positions, velocities, etc.
        config : SimulationConfig
            Simulation configuration
        output_dir : Path
            Base output directory for the simulation run
        cell_templates : Sequence[CellTemplate]
            Cell templates with mask centroid offsets
        rotated_masks : Optional[List[List[np.ndarray]]]
            Pre-computed rotated cell masks organised as [frame][cell]
        **kwargs : Any
            Additional arguments from CLI (as defined in get_cli_arguments)

        Returns
        -------
        ExportResult
            Result containing output path and optional metadata
        """
        ...


def discover_plugins() -> Dict[str, ExporterPlugin]:
    """
    Discover installed exporter plugins via entry points.

    Plugins register themselves in their pyproject.toml:
        [project.entry-points."rot_game.exporters"]
        plugin_name = "package.module:PluginClass"

    Returns
    -------
    Dict[str, ExporterPlugin]
        Dictionary mapping plugin names to plugin instances
    """
    plugins: Dict[str, ExporterPlugin] = {}

    try:
        # Python 3.10+ has importlib.metadata.entry_points with group parameter
        from importlib.metadata import entry_points

        eps = entry_points(group="rot_game.exporters")

        for ep in eps:
            try:
                plugin_class = ep.load()
                plugin_instance = plugin_class()

                if not isinstance(plugin_instance, ExporterPlugin):
                    logger.warning(
                        "Plugin '%s' does not implement ExporterPlugin interface, skipping",
                        ep.name,
                    )
                    continue

                plugins[plugin_instance.name] = plugin_instance
                logger.debug("Discovered exporter plugin: %s", plugin_instance.name)

            except Exception as e:
                logger.warning("Failed to load plugin '%s': %s", ep.name, e)

    except Exception as e:
        logger.debug("Plugin discovery failed: %s", e)

    return plugins


# Re-export trajectory plugin utilities for convenience
from ..core.trajectory import (
    TrajectoryGenerator,
    TrajectoryMetadata,
    register_trajectory_generator,
    unregister_trajectory_generator,
    get_registered_trajectory_types,
    get_trajectory_metadata,
    discover_trajectory_plugins,
)


def discover_all_plugins() -> None:
    """
    Discover all plugin types (exporters and trajectories).
    
    This should be called early in the application startup to ensure
    all custom trajectory types are registered before config loading.
    """
    discover_trajectory_plugins()
    # Exporter plugins are discovered on-demand via discover_plugins()


__all__ = [
    # Exporter plugins
    "CLIArgument",
    "ExportResult",
    "ExporterPlugin",
    "discover_plugins",
    # Trajectory plugins
    "TrajectoryGenerator",
    "TrajectoryMetadata",
    "register_trajectory_generator",
    "unregister_trajectory_generator",
    "get_registered_trajectory_types",
    "get_trajectory_metadata",
    "discover_trajectory_plugins",
    # Combined discovery
    "discover_all_plugins",
]
