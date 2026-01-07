"""
Asset loading utilities for the simulator.

This module converts the file paths defined in :mod:`cell_simulator.config`
into ready-to-use NumPy arrays for backgrounds, cell templates, and masks.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import CellConfig, SimulationConfig
from .core.scaling import radius_um_to_pixels


logger = logging.getLogger(__name__)

PRELOAD_BACKGROUND_THRESHOLD_BYTES = 400 * 1024 * 1024  # 400 MB default limit


class AssetLoadingError(RuntimeError):
    """Raised when an asset cannot be loaded or validated."""


@dataclass(frozen=True)
class CellTemplate:
    """Loaded cell template (scaled) and its binary mask."""

    id: str
    image: np.ndarray
    mask: np.ndarray
    template_path: Path
    mask_path: Path
    radius_um: float
    radius_px: float
    mask_centroid_offset: Tuple[float, float]  # (dx, dy) from image center to mask center

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _list_frame_files(directory: Path) -> List[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    unique_paths = {path.resolve() for path in candidates if path.exists()}
    return sorted(unique_paths, key=_natural_sort_key)


def _load_grayscale_image(path: Path) -> np.ndarray:
    """Load a grayscale image from disk, raising an error on failure."""
    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise AssetLoadingError(f"Failed to load image as grayscale: {path}")
    if array.ndim != 2:
        raise AssetLoadingError(f"Expected grayscale image but got shape {array.shape} for {path}")
    return array


def _load_mask(path: Path) -> np.ndarray:
    """
    Load a binary mask image and normalize it to a boolean array.

    Any non-zero pixel is treated as foreground. The result is a numpy array
    of dtype ``bool`` matching the mask shape.
    """
    mask_raw = _load_grayscale_image(path)
    mask_bool = mask_raw > 0
    if not mask_bool.any():
        raise AssetLoadingError(f"Mask contains no foreground pixels: {path}")
    return mask_bool


def _ping_pong_index(frame_idx: int, length: int) -> int:
    """Map a monotonically increasing index onto a ping-pong sequence of given length."""
    if length <= 1:
        return 0
    period = 2 * length - 2
    mod = frame_idx % period
    return mod if mod < length else period - mod


def _scale_background(
    image: np.ndarray,
    target_resolution: Tuple[int, int],
    magnification: float = 1.0,
    reference_magnification: float = 1.0
) -> np.ndarray:
    """Scale the background using an affine transform centred on the image.

    The behaviour matches Qt's view transform: we keep the original pixels, apply a
    centre-based scale, and sample with high-quality interpolation. This preserves
    brightness while delivering the expected zoom-in/zoom-out effect.
    """

    target_width, target_height = target_resolution
    height, width = image.shape

    # Ensure the background matches the target resolution prior to applying zoom
    if (width, height) != (target_width, target_height):
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape

    zoom_factor = float(magnification) / float(reference_magnification)

    if np.isclose(zoom_factor, 1.0, atol=1e-3):
        return image.astype(np.uint8)

    center_x = width / 2.0
    center_y = height / 2.0

    # Build affine transform equivalent to Qt's centred scaling
    transform = cv2.getRotationMatrix2D((center_x, center_y), 0.0, zoom_factor)

    interpolation = cv2.INTER_CUBIC if zoom_factor > 1.0 else cv2.INTER_AREA

    zoomed = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return zoomed.astype(np.uint8)


class BackgroundProvider:
    """Provide background frames (single image or sequence) with optional ping-pong looping."""

    def __init__(
        self,
        frame_paths: Sequence[Path],
        target_resolution: Tuple[int, int],
        magnification: float,
        reference_magnification: float,
        preload_threshold_bytes: int = PRELOAD_BACKGROUND_THRESHOLD_BYTES,
        cache_limit: int = 32,
    ):
        if not frame_paths:
            raise AssetLoadingError("No background frames provided")
        self._frame_paths = list(frame_paths)
        self._target_resolution = target_resolution
        self._magnification = magnification
        self._reference_magnification = reference_magnification
        self._cache_limit = max(1, cache_limit)
        self._preloaded_frames: Optional[List[np.ndarray]] = None
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._resized = False

        first_raw = _load_grayscale_image(self._frame_paths[0])
        first_scaled = _scale_background(
            first_raw,
            target_resolution,
            magnification,
            reference_magnification,
        )
        self._frame_shape = first_scaled.shape
        if first_scaled.shape != first_raw.shape:
            self._resized = True

        estimated_bytes = len(self._frame_paths) * first_scaled.size
        if estimated_bytes <= preload_threshold_bytes:
            frames: List[np.ndarray] = []
            for path in self._frame_paths:
                raw = _load_grayscale_image(path)
                scaled = _scale_background(
                    raw,
                    target_resolution,
                    magnification,
                    reference_magnification,
                )
                if scaled.shape != raw.shape:
                    self._resized = True
                frames.append(scaled.astype(np.uint8))
            self._preloaded_frames = frames
            logger.debug(
                "Preloaded %d background frames (%.2f MB)",
                len(frames),
                estimated_bytes / (1024 * 1024),
            )
        else:
            self._cache[0] = first_scaled.astype(np.uint8)
            logger.debug(
                "Streaming background frames (estimated %.2f MB > threshold %.2f MB)",
                estimated_bytes / (1024 * 1024),
                preload_threshold_bytes / (1024 * 1024),
            )

    @property
    def height(self) -> int:
        return self._frame_shape[0]

    @property
    def width(self) -> int:
        return self._frame_shape[1]

    @property
    def source_frame_count(self) -> int:
        return len(self._frame_paths)

    @property
    def resized(self) -> bool:
        return self._resized

    @property
    def is_preloaded(self) -> bool:
        return self._preloaded_frames is not None

    def _map_index(self, frame_idx: int) -> int:
        count = len(self._frame_paths)
        if count <= 2:
            return frame_idx % count
        return _ping_pong_index(frame_idx, count)

    def frame_for(self, frame_idx: int) -> np.ndarray:
        mapped_idx = self._map_index(frame_idx)

        if self._preloaded_frames is not None:
            return self._preloaded_frames[mapped_idx]

        if mapped_idx in self._cache:
            frame = self._cache.pop(mapped_idx)
            self._cache[mapped_idx] = frame
            return frame

        path = self._frame_paths[mapped_idx]
        raw = _load_grayscale_image(path)
        scaled = _scale_background(
            raw,
            self._target_resolution,
            self._magnification,
            self._reference_magnification,
        )
        if scaled.shape != raw.shape:
            self._resized = True

        frame = scaled.astype(np.uint8)
        self._cache[mapped_idx] = frame
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return frame



def _scale_template_to_radius(
    image: np.ndarray,
    mask: np.ndarray,
    target_radius_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale template and mask so that radius matches the requested size."""

    if target_radius_px <= 0:
        return image, mask

    # Diameter based on desired radius (ensure at least 1 pixel)
    target_diameter = max(int(round(target_radius_px * 2.0)), 1)

    orig_height, orig_width = image.shape
    max_dim = max(orig_width, orig_height)
    if max_dim <= 0:
        return image, mask

    scale = target_diameter / max_dim
    if np.isclose(scale, 1.0, atol=1e-3):
        return image, mask

    new_width = max(1, int(round(orig_width * scale)))
    new_height = max(1, int(round(orig_height * scale)))

    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    scaled_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST) > 0

    return scaled_image.astype(np.uint8), scaled_mask


def _compute_mask_centroid_offset(mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute the offset from the image center to the mask's center of mass.
    
    This offset is used to ensure that the mask centroid (not the image center)
    is positioned at trajectory points and used as the rotation pivot.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask array (bool or uint8)
        
    Returns
    -------
    Tuple[float, float]
        Offset (dx, dy) from image center to mask center of mass
    """
    if not mask.any():
        # No foreground pixels - return zero offset (shouldn't happen with validation)
        return (0.0, 0.0)
    
    # Compute mask center of mass
    y_coords, x_coords = np.where(mask)
    mask_center_x = float(np.mean(x_coords))
    mask_center_y = float(np.mean(y_coords))
    
    # Compute image center
    height, width = mask.shape
    image_center_x = width / 2.0
    image_center_y = height / 2.0
    
    # Return offset from image center to mask center
    offset_x = mask_center_x - image_center_x
    offset_y = mask_center_y - image_center_y
    
    return (offset_x, offset_y)


def _pad_for_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    mask_centroid_offset: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Pad the template to prevent clipping during rotation around mask centroid.
    
    When rotating around an off-center point, parts of the image can extend beyond
    the original bounds. This function calculates the minimum padding needed to
    contain a full 360° rotation without clipping.
    
    Parameters
    ----------
    image : np.ndarray
        Template image
    mask : np.ndarray
        Template mask
    mask_centroid_offset : Tuple[float, float]
        Offset (dx, dy) from image center to mask centroid
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Tuple[float, float]]
        Padded image, padded mask, and mask centroid offset (unchanged)
    """
    height, width = image.shape
    offset_x, offset_y = mask_centroid_offset
    
    # Rotation center in image coordinates
    rotation_center_x = width / 2.0 + offset_x
    rotation_center_y = height / 2.0 + offset_y
    
    # Find maximum distance from rotation center to any corner
    corners = [
        (0.0, 0.0),
        (float(width), 0.0),
        (0.0, float(height)),
        (float(width), float(height)),
    ]
    
    max_distance = 0.0
    for corner_x, corner_y in corners:
        dist = np.sqrt((corner_x - rotation_center_x) ** 2 + (corner_y - rotation_center_y) ** 2)
        max_distance = max(max_distance, dist)
    
    # Required canvas size to contain full rotation (diameter = 2 * radius)
    required_size = int(np.ceil(max_distance * 2.0))
    
    # Calculate padding needed (symmetric on all sides)
    pad_width = max(0, (required_size - width) // 2)
    pad_height = max(0, (required_size - height) // 2)
    
    # Add a small safety margin (2 pixels) to handle floating point rounding
    pad_width += 2
    pad_height += 2
    
    # If padding is minimal (< 3 pixels), skip padding to save memory
    if pad_width < 3 and pad_height < 3:
        return image, mask, mask_centroid_offset
    
    # Pad image and mask symmetrically
    padded_image = np.pad(
        image,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode='constant',
        constant_values=0,
    )
    
    padded_mask = np.pad(
        mask,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode='constant',
        constant_values=False,
    )
    
    # The mask_centroid_offset remains the same because both the mask and image
    # center move by the same amount (pad_width, pad_height)
    return padded_image, padded_mask, mask_centroid_offset


def _load_cell_template(cell_config: CellConfig, magnification: float, pixel_size_um: float) -> CellTemplate:
    """Load, validate, and scale a cell template along with its mask."""
    image = _load_grayscale_image(cell_config.template)
    mask = _load_mask(cell_config.mask.path)

    if image.shape != mask.shape:
        raise AssetLoadingError(
            f"Template and mask shapes differ for cell '{cell_config.id}': "
            f"{image.shape} vs {mask.shape}"
        )

    radius_px = radius_um_to_pixels(
        radius_um=float(cell_config.radius_um),
        magnification=magnification,
        pixel_size_um=pixel_size_um,
    )

    scaled_image, scaled_mask = _scale_template_to_radius(image, mask, radius_px)

    # Compute mask centroid offset from image center
    mask_centroid_offset = _compute_mask_centroid_offset(scaled_mask)

    # Pad template to prevent clipping during rotation around mask centroid
    # This is done once at load time for optimal performance
    padded_image, padded_mask, mask_centroid_offset = _pad_for_rotation(
        scaled_image,
        scaled_mask,
        mask_centroid_offset,
    )

    return CellTemplate(
        id=cell_config.id,
        image=padded_image,
        mask=padded_mask,
        template_path=cell_config.template,
        mask_path=cell_config.mask.path,
        radius_um=float(cell_config.radius_um),
        radius_px=float(radius_px),
        mask_centroid_offset=mask_centroid_offset,
    )


class AssetManager:
    """Loads simulator assets defined by a :class:`SimulationConfig`."""

    def __init__(self, config: SimulationConfig):
        self._config = config

    def load_background(self):
        """Load the configured background as a unified provider."""
        video_cfg = self._config.video
        magnification = float(video_cfg.magnification)
        ref_magnification = float(getattr(video_cfg, "background_ref_mag", 10.93))
        target_resolution = tuple(video_cfg.resolution)

        frame_paths: List[Path] = []
        if video_cfg.background is not None:
            frame_paths = [video_cfg.background]
        elif video_cfg.background_frames_dir is not None:
            frame_paths = _list_frame_files(video_cfg.background_frames_dir)
        if not frame_paths:
            raise AssetLoadingError("No background source configured")

        return BackgroundProvider(
            frame_paths=frame_paths,
            target_resolution=target_resolution,
            magnification=magnification,
            reference_magnification=ref_magnification,
            preload_threshold_bytes=PRELOAD_BACKGROUND_THRESHOLD_BYTES,
        )

    def load_cell_templates(self) -> List[CellTemplate]:
        """Load all cell templates specified in the scenario."""
        magnification = float(self._config.video.magnification)
        pixel_size = float(self._config.video.pixel_size_um)
        return [
            _load_cell_template(cell_conf, magnification, pixel_size)
            for cell_conf in self._config.cells
        ]
