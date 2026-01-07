"""Trajectory generation utilities.

This module converts the declarative trajectory configuration into
per-frame positions for each cell, respecting the simulator's
requirements (movement only during active intervals, support for
multiple trajectory types, etc.).

The module also provides a plugin registry for custom trajectory types.
External plugins can register new trajectory generators via the
`register_trajectory_generator` function or the `rot_game.trajectories`
entry point group.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

import numpy as np

from ..config import SimulationConfig, TrajectoryConfig
from .schedule import FrameSchedule

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory Plugin Registry
# ---------------------------------------------------------------------------

class TrajectoryGenerator(Protocol):
    """Protocol for trajectory generator functions.
    
    Trajectory generators receive the trajectory configuration and frame count,
    and return an array of (x, y) positions for each frame.
    
    Parameters
    ----------
    traj_conf : TrajectoryConfig
        The trajectory configuration from the scenario YAML.
    total_active_frames : int
        Number of active frames to generate positions for.
    fps : float
        Frames per second of the video (useful for time-based calculations).
    **kwargs : Any
        Additional parameters that may be passed by the trajectory system.
    
    Returns
    -------
    np.ndarray
        Array of shape (total_active_frames, 2) containing (x, y) positions.
    """
    
    def __call__(
        self,
        traj_conf: TrajectoryConfig,
        total_active_frames: int,
        fps: float = 30.0,
        **kwargs: Any,
    ) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Trajectory Metadata
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryMetadata:
    """Metadata describing UI hints and requirements for a trajectory type.
    
    Plugins can provide metadata to customize how the GUI displays
    trajectory options without hardcoding trajectory-specific logic.
    
    Attributes
    ----------
    start_label : str
        Label for the start position (default: "Start")
    end_label : str
        Label for the end position (default: "End")
    show_end : bool
        Whether to show end position controls (default: True)
    show_controls : bool
        Whether to show Bézier control points (default: True)
    num_control_points : int
        Number of control points (0, 1, or 2)
    description : str
        Human-readable description of the trajectory type
    params_schema : Dict[str, Dict[str, Any]]
        Schema for custom parameters. Each key is a parameter name,
        and the value is a dict with keys: 'type' (float/int/choice),
        'default', 'label', and optional 'min', 'max', 'options'.
        Example: {'frequency': {'type': 'float', 'default': 1.0, 
                                'min': 0.01, 'max': 100, 'label': 'Frequency (Hz)'}}
    shared_params : bool
        If True, parameter changes propagate to all cells with this trajectory type.
    custom_widget_factory : Callable, optional
        A factory function that creates a custom QWidget for additional UI.
        Signature: (cell_data: dict, update_callback: Callable) -> QWidget
        The cell_data contains 'start', 'end', 'params', 'magnification'.
        The update_callback should be called with (field_name, value) when values change.
    """
    start_label: str = "Start"
    end_label: str = "End"
    show_end: bool = True
    show_controls: bool = True
    num_control_points: int = 0
    description: str = ""
    params_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    shared_params: bool = False
    custom_widget_factory: Optional[Callable[..., Any]] = None


# Built-in trajectory metadata
_builtin_metadata: Dict[str, TrajectoryMetadata] = {
    "linear": TrajectoryMetadata(
        start_label="Start",
        end_label="End",
        show_end=True,
        show_controls=False,
        num_control_points=0,
        description="Linear motion from start to end",
    ),
    "parabolic": TrajectoryMetadata(
        start_label="Start",
        end_label="End",
        show_end=True,
        show_controls=True,
        num_control_points=1,
        description="Quadratic Bézier curve with one control point",
    ),
    "cubic": TrajectoryMetadata(
        start_label="Start",
        end_label="End",
        show_end=True,
        show_controls=True,
        num_control_points=2,
        description="Cubic Bézier curve with two control points",
    ),
    "stationary": TrajectoryMetadata(
        start_label="Position",
        end_label="End",
        show_end=False,
        show_controls=False,
        num_control_points=0,
        description="Cell remains at a fixed position",
    ),
}


# Registry for custom trajectory generators and metadata
_trajectory_registry: Dict[str, TrajectoryGenerator] = {}
_trajectory_metadata: Dict[str, TrajectoryMetadata] = {}


def register_trajectory_generator(
    name: str,
    generator: TrajectoryGenerator,
    *,
    metadata: Optional[TrajectoryMetadata] = None,
    override: bool = False,
) -> None:
    """
    Register a custom trajectory generator function.
    
    This allows external plugins to add new trajectory types without
    modifying the core codebase.
    
    Parameters
    ----------
    name : str
        The trajectory type name (used in YAML config as `type: <name>`).
    generator : TrajectoryGenerator
        A callable that generates trajectory positions.
    metadata : TrajectoryMetadata, optional
        UI metadata for the trajectory type. If not provided, default
        metadata will be used.
    override : bool
        If True, allow overriding existing generators. Default False.
    
    Raises
    ------
    ValueError
        If a generator with the same name already exists and override is False.
    
    Example
    -------
    >>> def my_custom_trajectory(traj_conf, total_active_frames, fps=30.0, **kwargs):
    ...     # Generate positions...
    ...     return positions
    >>> register_trajectory_generator(
    ...     "my_custom",
    ...     my_custom_trajectory,
    ...     metadata=TrajectoryMetadata(start_label="Center", show_controls=False),
    ... )
    """
    if name in _trajectory_registry and not override:
        raise ValueError(
            f"Trajectory generator '{name}' already registered. "
            "Use override=True to replace it."
        )
    
    _trajectory_registry[name] = generator
    _trajectory_metadata[name] = metadata or TrajectoryMetadata()
    logger.debug("Registered trajectory generator: %s", name)


def unregister_trajectory_generator(name: str) -> bool:
    """
    Unregister a custom trajectory generator.
    
    Parameters
    ----------
    name : str
        The trajectory type name to unregister.
    
    Returns
    -------
    bool
        True if the generator was removed, False if it wasn't registered.
    """
    if name in _trajectory_registry:
        del _trajectory_registry[name]
        _trajectory_metadata.pop(name, None)
        logger.debug("Unregistered trajectory generator: %s", name)
        return True
    return False


def get_registered_trajectory_types() -> List[str]:
    """
    Get a list of all registered custom trajectory types.
    
    Returns
    -------
    List[str]
        Names of registered trajectory generators.
    """
    return list(_trajectory_registry.keys())


def get_trajectory_metadata(trajectory_type: str) -> TrajectoryMetadata:
    """
    Get UI metadata for a trajectory type.
    
    Parameters
    ----------
    trajectory_type : str
        The trajectory type name.
    
    Returns
    -------
    TrajectoryMetadata
        Metadata for the trajectory type. Falls back to defaults
        if the type is not found.
    """
    # Check plugin registry first
    if trajectory_type in _trajectory_metadata:
        return _trajectory_metadata[trajectory_type]
    
    # Check built-in metadata
    if trajectory_type in _builtin_metadata:
        return _builtin_metadata[trajectory_type]
    
    # Return default metadata for unknown types
    return TrajectoryMetadata()


def discover_trajectory_plugins() -> None:
    """
    Discover and load trajectory plugins via entry points.
    
    Plugins register themselves in their pyproject.toml:
        [project.entry-points."rot_game.trajectories"]
        plugin_name = "package.module:register_function"
    
    The entry point should point to a function that calls
    `register_trajectory_generator` when invoked.
    """
    try:
        from importlib.metadata import entry_points
        
        eps = entry_points(group="rot_game.trajectories")
        
        for ep in eps:
            try:
                register_func = ep.load()
                # Call the registration function
                register_func()
                logger.debug("Loaded trajectory plugin: %s", ep.name)
            except Exception as e:
                logger.warning("Failed to load trajectory plugin '%s': %s", ep.name, e)
    
    except Exception as e:
        logger.debug("Trajectory plugin discovery failed: %s", e)


# ---------------------------------------------------------------------------
# Built-in Trajectory Generators
# ---------------------------------------------------------------------------


def _generate_linear_trajectory(traj_conf: TrajectoryConfig, total_active_frames: int) -> np.ndarray:
    start = np.array(traj_conf.start, dtype=np.float64)
    end = np.array(traj_conf.end, dtype=np.float64)
    # Include both endpoints by generating total_active_frames+1 points, then drop the last
    t_values = np.linspace(0.0, 1.0, num=total_active_frames, endpoint=True)
    return start + np.outer(t_values, (end - start))


def _generate_parabolic_trajectory(traj_conf: TrajectoryConfig, total_active_frames: int) -> np.ndarray:
    start = np.array(traj_conf.start, dtype=np.float64)
    end = np.array(traj_conf.end, dtype=np.float64)
    control = np.array(traj_conf.control_points[0], dtype=np.float64)
    t_values = np.linspace(0.0, 1.0, num=total_active_frames, endpoint=True)
    positions = (1 - t_values)[:, None] ** 2 * start + 2 * (1 - t_values)[:, None] * t_values[:, None] * control + t_values[:, None] ** 2 * end
    return positions


def _generate_cubic_trajectory(traj_conf: TrajectoryConfig, total_active_frames: int) -> np.ndarray:
    start = np.array(traj_conf.start, dtype=np.float64)
    end = np.array(traj_conf.end, dtype=np.float64)
    control1 = np.array(traj_conf.control_points[0], dtype=np.float64)
    control2 = np.array(traj_conf.control_points[1], dtype=np.float64)
    t_values = np.linspace(0.0, 1.0, num=total_active_frames, endpoint=True)
    positions = (
        (1 - t_values)[:, None] ** 3 * start
        + 3 * (1 - t_values)[:, None] ** 2 * t_values[:, None] * control1
        + 3 * (1 - t_values)[:, None] * t_values[:, None] ** 2 * control2
        + t_values[:, None] ** 3 * end
    )
    return positions


def _generate_stationary_trajectory(
    traj_conf: TrajectoryConfig,
    total_active_frames: int,
    fps: float = 30.0,
    **kwargs: Any,
) -> np.ndarray:
    start = np.array(traj_conf.start, dtype=np.float64)
    return np.repeat(start[None, :], repeats=total_active_frames, axis=0)


def _generate_path(
    traj_conf: TrajectoryConfig,
    total_active_frames: int,
    fps: float = 30.0,
    **kwargs: Any,
) -> np.ndarray:
    if total_active_frames <= 0:
        # No active motion; return a single position at the start as a placeholder
        return np.array([traj_conf.start], dtype=np.float64)

    # Built-in trajectory generators
    builtin_generators = {
        "linear": _generate_linear_trajectory,
        "parabolic": _generate_parabolic_trajectory,
        "cubic": _generate_cubic_trajectory,
        "stationary": _generate_stationary_trajectory,
    }

    # Check custom registry first (allows overriding built-ins if needed)
    if traj_conf.type in _trajectory_registry:
        generator = _trajectory_registry[traj_conf.type]
        path = generator(traj_conf, total_active_frames, fps=fps, **kwargs)
        return np.asarray(path, dtype=np.float64)

    # Fall back to built-in generators
    if traj_conf.type not in builtin_generators:
        available = list(builtin_generators.keys()) + list(_trajectory_registry.keys())
        raise ValueError(
            f"Unsupported trajectory type: '{traj_conf.type}'. "
            f"Available types: {available}"
        )

    if traj_conf.type == "parabolic":
        if not traj_conf.control_points or len(traj_conf.control_points) < 1:
            raise ValueError("Parabolic trajectory requires at least one control point")
    if traj_conf.type == "cubic":
        if not traj_conf.control_points or len(traj_conf.control_points) != 2:
            raise ValueError("Cubic trajectory requires exactly two control points")

    path = builtin_generators[traj_conf.type](traj_conf, total_active_frames)
    return np.asarray(path, dtype=np.float64)


def build_per_frame_positions(
    config: SimulationConfig,
    frame_schedule: FrameSchedule,
) -> List[np.ndarray]:
    """Generate per-frame positions for each cell based on the frame schedule."""

    total_frames = frame_schedule.total_frames
    total_active_frames = frame_schedule.total_active_frames
    fps = float(config.video.fps)

    if total_active_frames <= 0:
        raise ValueError("Schedule must contain at least one active frame for trajectory generation")

    results: List[np.ndarray] = []

    for cell_config in config.cells:
        path = _generate_path(cell_config.trajectory, total_active_frames, fps=fps)

        positions = np.zeros((total_frames, 2), dtype=np.float64)
        frame_idx = 0
        path_idx = 0

        for active_frames, delay_frames in zip(frame_schedule.active_frame_counts, frame_schedule.delay_frame_counts):
            # Active frames: advance along the trajectory path
            for _ in range(active_frames):
                current_position = path[path_idx]
                positions[frame_idx] = current_position
                frame_idx += 1
                path_idx += 1

            # Delay frames: hold the last active position
            if delay_frames > 0:
                hold_position = positions[frame_idx - 1] if frame_idx > 0 else path[min(path_idx, path.shape[0] - 1)]
                for _ in range(delay_frames):
                    positions[frame_idx] = hold_position
                    frame_idx += 1

        # Fill any trailing frames (due to rounding) with the final position
        if frame_idx < total_frames:
            fill_value = positions[frame_idx - 1] if frame_idx > 0 else path[-1]
            positions[frame_idx:] = fill_value

        results.append(positions)

    return results
