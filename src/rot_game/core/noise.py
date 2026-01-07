"""
Noise simulation for realistic microscopy imaging.

This module provides utilities to add realistic camera noise to simulated
frames, mimicking quantization errors and read noise common in microscopy.
"""

from __future__ import annotations

import numpy as np


def add_quantization_noise(frame: np.ndarray, stddev: float) -> np.ndarray:
    """
    Add Gaussian noise to simulate camera quantization and read noise.
    
    This simulates the signal-independent read noise present in all camera systems,
    which becomes particularly visible in microscopy imaging. The noise follows a
    Gaussian distribution centered at zero.
    
    Parameters
    ----------
    frame : np.ndarray
        Input frame (grayscale, dtype uint8 or float32)
    stddev : float
        Standard deviation of the Gaussian noise. Typical values:
        - 2-5: Subtle noise (high-quality cameras)
        - 5-10: Moderate noise (standard cameras)
        - 10-20: Heavy noise (low-light or older cameras)
        
    Returns
    -------
    np.ndarray
        Noisy frame (dtype uint8), with values clipped to [0, 255]
        
    Examples
    --------
    >>> frame = np.ones((100, 100), dtype=np.uint8) * 128
    >>> noisy = add_quantization_noise(frame, stddev=5.0)
    >>> noisy.dtype
    dtype('uint8')
    """
    if stddev <= 0:
        return frame.astype(np.uint8)
    
    # Convert to float for noise addition
    frame_float = frame.astype(np.float32)
    
    # Generate Gaussian noise with specified standard deviation
    noise = np.random.normal(0, stddev, frame.shape).astype(np.float32)
    
    # Add noise and clip to valid uint8 range
    noisy_frame = frame_float + noise
    noisy_frame = np.clip(noisy_frame, 0, 255)
    
    return noisy_frame.astype(np.uint8)


def add_gaussian_noise(frame: np.ndarray, stddev: float) -> np.ndarray:
    """
    Alias for add_quantization_noise for clarity in different contexts.
    
    See add_quantization_noise for full documentation.
    """
    return add_quantization_noise(frame, stddev)
