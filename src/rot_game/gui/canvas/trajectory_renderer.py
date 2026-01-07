"""Pure computational logic for trajectory path generation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...core.trajectory import _generate_path
from ...config import TrajectoryConfig
from ..models import ScenarioCell


class TrajectoryRenderer:
    """Renders trajectory paths for visualization.
    
    This class provides pure computational logic for generating trajectory paths
    without any UI dependencies, making it fully testable.
    """
    
    @staticmethod
    def trajectory_path(
        cell: ScenarioCell,
        samples: int = 200,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate trajectory path coordinates for visualization.
        
        Args:
            cell: The cell with trajectory configuration
            samples: Number of points to sample along the trajectory
            
        Returns:
            Tuple of (x_coords, y_coords) arrays, or None if trajectory is invalid
        """
        config = TrajectoryConfig(
            type=cell.trajectory_type,
            start=cell.start,
            end=cell.end,
            control_points=cell.control_points if cell.control_points else None,
            params=getattr(cell, 'params', None) or None,
        )
        
        try:
            trajectory = _generate_path(config, total_active_frames=samples)
            return trajectory[:, 0], trajectory[:, 1]
        except ValueError:
            # Invalid trajectory configuration
            return None
