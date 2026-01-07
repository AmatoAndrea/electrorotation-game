"""Utilities for calculating cell radius from mask images."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


# Template pixel size in micrometers
TEMPLATE_PIXEL_SIZE_UM = 0.677


def calculate_radius_from_mask(mask_path: Path) -> float:
    """Calculate cell radius from a binary mask using area-based method.
    
    The radius is computed as r = sqrt(area/π), representing the radius of
    a circle with equivalent area to the mask's foreground pixels.
    
    Args:
        mask_path: Path to the binary mask image
        
    Returns:
        Cell radius in micrometers
        
    Raises:
        FileNotFoundError: If mask file doesn't exist
        ValueError: If mask is empty or invalid
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load mask as grayscale
    mask_array = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_array is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    
    # Count foreground pixels (any non-zero value)
    mask_area_pixels = np.count_nonzero(mask_array)
    if mask_area_pixels == 0:
        raise ValueError(f"Mask is empty (no foreground pixels): {mask_path}")
    
    # Calculate equivalent circle radius in pixels
    radius_pixels = math.sqrt(mask_area_pixels / math.pi)
    
    # Convert to micrometers
    radius_um = radius_pixels * TEMPLATE_PIXEL_SIZE_UM
    
    return radius_um
