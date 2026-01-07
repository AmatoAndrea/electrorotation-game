"""
Frame rendering for the simulator.

The renderer composes simulated frames from the background image,
per-cell templates, and the frame-level schedule/trajectory data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union, Tuple, Optional

import cv2
import numpy as np

from ..assets import CellTemplate
from ..config import SimulationConfig
from .feather import FeatherParameters, signed_distance_alpha
from .schedule import FrameSchedule
from .trajectory import build_per_frame_positions
from .scaling import radius_um_to_pixels
from .noise import add_quantization_noise
from .noise import add_quantization_noise


@dataclass
class CellGroundTruth:
    """Ground-truth information for a simulated cell."""

    cell_id: str
    positions: np.ndarray  # shape (frames, 2)
    angular_velocity_rad_s: np.ndarray  # per frame
    cumulative_angle_rad: np.ndarray  # per frame
    radius_um: float
    radius_px: float
    trajectory_type: str
    area_px: np.ndarray  # per frame

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "positions": self.positions.tolist(),
            "angular_velocity_rad_s": self.angular_velocity_rad_s.tolist(),
            "cumulative_angle_rad": self.cumulative_angle_rad.tolist(),
            "radius_um": self.radius_um,
            "radius_px": self.radius_px,
            "trajectory_type": self.trajectory_type,
        }


@dataclass
class GroundTruth:
    """Aggregate ground-truth payload for the entire simulation."""

    cells: List[CellGroundTruth]
    fps: float

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "cells": [cell.to_dict() for cell in self.cells],
        }


class Renderer:
    """
    Compose simulator frames from configuration, assets, schedule, and trajectories.

    Parameters
    ----------
    config:
        Simulation configuration.
    background:
        Loaded background image.
    cell_templates:
        Loaded cell templates.
    frame_schedule:
        Frame-level schedule describing angular velocity and delays.
    """

    def __init__(
        self,
        config: SimulationConfig,
        background,
        cell_templates: Sequence[CellTemplate],
        frame_schedule: FrameSchedule,
    ):
        self._config = config
        self._background = background
        self._cell_templates = list(cell_templates)
        self._schedule = frame_schedule

        if len(self._cell_templates) != len(self._config.cells):
            raise ValueError("Number of loaded cell templates must match configuration")

        bg_height, bg_width = self._background.height, self._background.width
        video_width, video_height = self._config.video.resolution
        if (bg_width, bg_height) != (video_width, video_height):
            raise ValueError(
                "Background resolution mismatch: "
                f"background {(bg_width, bg_height)}, expected {(video_width, video_height)}"
            )

        expected_frames = int(round(self._config.video.duration_s * self._config.video.fps))
        if self._schedule.total_frames != expected_frames:
            raise ValueError(
                "Schedule frame count does not match video duration: "
                f"{self._schedule.total_frames} vs {expected_frames}"
            )

        self._positions = build_per_frame_positions(self._config, self._schedule)
        self._fps = float(self._config.video.fps)
        self._angular_velocity_rad_per_frame = self._schedule.angular_velocity_rad_per_frame
        self._cumulative_angle_rad = np.cumsum(self._angular_velocity_rad_per_frame)
        self._noise_enabled = bool(self._config.video.noise_enabled)
        self._noise_stddev = float(self._config.video.noise_stddev)
        self._num_cells = len(self._config.cells)

    def compute_ground_truth(self) -> GroundTruth:
        """Build the ground-truth payload without rendering frames."""
        ground_truth_cells: List[CellGroundTruth] = []
        angular_velocity_rad_s = self._angular_velocity_rad_per_frame * self._fps

        for cell_config, positions in zip(self._config.cells, self._positions):
            radius_px = radius_um_to_pixels(
                radius_um=float(cell_config.radius_um),
                magnification=float(self._config.video.magnification),
                pixel_size_um=float(self._config.video.pixel_size_um),
            )

            ground_truth_cells.append(
                CellGroundTruth(
                    cell_id=cell_config.id,
                    positions=positions,
                    angular_velocity_rad_s=angular_velocity_rad_s,
                    cumulative_angle_rad=self._cumulative_angle_rad,
                    radius_um=float(cell_config.radius_um),
                    radius_px=radius_px,
                    trajectory_type=cell_config.trajectory.type,
                    area_px=np.zeros_like(angular_velocity_rad_s, dtype=np.int32),
                )
            )

        return GroundTruth(cells=ground_truth_cells, fps=self._fps)

    def render(
        self,
        feather_pixels: int = 0,
        capture_masks: bool = False,
        feather_params: Optional[FeatherParameters] = None,
    ) -> Union[
        Tuple[np.ndarray, GroundTruth],
        Tuple[np.ndarray, GroundTruth, list[list[np.ndarray]]],
    ]:
        """
        Render the full video sequence.

        Parameters
        ----------
        feather_pixels:
            Optional erosion of the mask edges (in pixels) to reduce seams.
            A value of 0 keeps binary masks.
        capture_masks:
            If True, also capture and return rotated cell masks for each frame.
            This enables tracking-compatible export.
        feather_params:
            Optional signed-distance feathering settings. When provided, mask
            alpha is derived from a signed distance field using the requested
            inside/outside widths. Falls back to binary alpha when omitted.

        Returns
        -------
        frames:
            Array of shape (frames, height, width) in uint8.
        ground_truth:
            GroundTruth object describing true trajectories and rotations.
        rotated_masks: (optional, if capture_masks=True)
            Array of shape (frames, num_cells, mask_height, mask_width) containing
            the rotated masks for each cell at each frame.
        """

        bg_height = self._background.height
        bg_width = self._background.width

        frames = np.empty((self._schedule.total_frames, bg_height, bg_width), dtype=np.uint8)
        for frame_idx in range(self._schedule.total_frames):
            frames[frame_idx] = self._background.frame_for(frame_idx)

        ground_truth = self.compute_ground_truth()

        # Initialize mask storage if requested
        rotated_masks: Optional[list[list[Optional[np.ndarray]]]] = None
        if capture_masks:
            num_cells = len(ground_truth.cells)
            rotated_masks = [
                [None for _ in range(num_cells)]
                for _ in range(self._schedule.total_frames)
            ]

        for cell_idx, (cell_gt, template) in enumerate(zip(ground_truth.cells, self._cell_templates)):
            positions = cell_gt.positions

            # Precompute feathered alpha once per template to avoid per-frame SDF recomputation
            mask_bool = template.mask.astype(bool)
            image_uint8 = template.image.astype(np.uint8)
            binary_mask_uint8 = mask_bool.astype(np.uint8) * 255

            if feather_params and feather_params.is_active():
                alpha_float = signed_distance_alpha(mask_bool, feather_params)
            else:
                alpha_float = mask_bool.astype(np.float32)

            alpha_uint8 = np.clip(alpha_float * 255.0, 0, 255).astype(np.uint8)

            height, width = image_uint8.shape
            base_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            base_rgba[..., 0] = image_uint8
            base_rgba[..., 1] = image_uint8
            base_rgba[..., 2] = image_uint8
            base_rgba[..., 3] = alpha_uint8

            for frame_idx in range(self._schedule.total_frames):
                # OpenCV rotates clockwise for positive angles (screen-space), so flip sign
                angle_degrees = -np.degrees(cell_gt.cumulative_angle_rad[frame_idx])
                offset_x, offset_y = template.mask_centroid_offset
                target_x = float(positions[frame_idx][0] - offset_x)
                target_y = float(positions[frame_idx][1] - offset_y)
                center_x = int(round(target_x))
                center_y = int(round(target_y))
                frac_x = target_x - center_x
                frac_y = target_y - center_y

                rotated_template, rotated_mask, rotated_binary_mask = _rotate_template_and_mask(
                    base_rgba,
                    binary_mask_uint8,
                    angle_degrees,
                    template.mask_centroid_offset,
                    feather_pixels,
                    frac_x=frac_x,
                    frac_y=frac_y,
                )

                if self._noise_enabled:
                    # Apply noise only within the cell mask
                    mask_bool = rotated_mask > 0.0
                    if mask_bool.any():
                        noisy = add_quantization_noise(rotated_template, self._noise_stddev).astype(np.float32)
                        rotated_template = rotated_template.copy()
                        rotated_template[mask_bool] = noisy[mask_bool]

                # Capture mask if requested
                if capture_masks and rotated_masks is not None:
                    rotated_masks[frame_idx][cell_idx] = rotated_binary_mask
                
                frames[frame_idx] = _composite_cell(
                    base_frame=frames[frame_idx],
                    cell_image=rotated_template,
                    cell_mask=rotated_mask,
                    binary_mask=rotated_binary_mask,
                    position=np.array([center_x, center_y], dtype=np.float32),
                    mask_centroid_offset=template.mask_centroid_offset,
                    area_store=cell_gt.area_px,
                    frame_idx=frame_idx,
                )

        if capture_masks and rotated_masks is not None:
            # Cast away optional type (should be filled during rendering)
            filled_masks: list[list[np.ndarray]] = [
                [mask if mask is not None else np.zeros_like(self._cell_templates[idx].mask, dtype=np.float32)
                 for idx, mask in enumerate(frame_masks)]
                for frame_masks in rotated_masks
            ]
            return frames, ground_truth, filled_masks
        return frames, ground_truth


def _rotate_template_and_mask(
    base_rgba: np.ndarray,
    binary_mask_uint8: np.ndarray,
    angle_degrees: float,
    mask_centroid_offset: Tuple[float, float],
    feather_pixels: int,
    frac_x: float = 0.0,
    frac_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate a precomputed RGBA template (with feathered alpha) plus binary mask.
    
    Mimics Qt's approach: rotates image and alpha together with smooth interpolation,
    then separates. This keeps preview/export parity without per-frame SDF recompute.
    
    The rotation now pivots around the mask centroid rather than the image center,
    ensuring that asymmetric cell templates rotate correctly around their actual center.
    """

    height, width, _ = base_rgba.shape
    
    # Compute rotation center: image center + mask centroid offset
    image_center_x = width / 2.0
    image_center_y = height / 2.0
    offset_x, offset_y = mask_centroid_offset
    rotation_center = (image_center_x + offset_x, image_center_y + offset_y)
    
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle_degrees, 1.0)
    # Incorporate subpixel translation to reduce jitter
    if abs(frac_x) > 1e-3 or abs(frac_y) > 1e-3:
        rotation_matrix[0, 2] += frac_x
        rotation_matrix[1, 2] += frac_y
    
    # Rotate the RGBA image as a single unit (like Qt does!)
    rotated_rgba = cv2.warpAffine(
        base_rgba,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,  # Smooth interpolation for all channels
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Also rotate the binary mask with nearest neighbor for clean area accounting
    rotated_binary_mask = cv2.warpAffine(
        binary_mask_uint8,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    
    # Extract rotated grayscale and alpha
    rotated_image = rotated_rgba[..., 0].astype(np.float32)
    rotated_mask_uint8 = rotated_rgba[..., 3]

    if feather_pixels > 0:
        kernel_size = max(1, feather_pixels * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        rotated_mask_uint8 = cv2.erode(rotated_mask_uint8, kernel, iterations=1)

    rotated_mask_alpha = rotated_mask_uint8.astype(np.float32) / 255.0

    return rotated_image, rotated_mask_alpha, rotated_binary_mask.astype(np.uint8)


def _composite_cell(
    base_frame: np.ndarray,
    cell_image: np.ndarray,
    cell_mask: np.ndarray,
    binary_mask: np.ndarray,
    position: np.ndarray,
    mask_centroid_offset: Tuple[float, float],
    area_store: Optional[np.ndarray] = None,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Composite a single cell onto the base frame using alpha blending.
    
    This matches Qt's alpha blending behavior for smooth, anti-aliased edges.
    
    The position parameter specifies where the mask centroid should be placed.
    The mask_centroid_offset is used to adjust the image placement so that the
    mask centroid (not the image center) ends up at the specified position.
    
    Parameters
    ----------
    base_frame:
        Background frame (uint8)
    cell_image:
        Rotated cell template (float32)
    cell_mask:
        Rotated mask as alpha channel (float32, range 0.0-1.0)
    position:
        Center position (x, y) in pixels - where the mask centroid should be placed
    mask_centroid_offset:
        Offset (dx, dy) from image center to mask centroid
    """

    result = base_frame.astype(np.float32)  # Work in float for precision
    template_h, template_w = cell_image.shape

    # Adjust position: the trajectory position should map to the mask centroid,
    # but we need to position the image. Since the mask centroid is offset from
    # the image center, we need to subtract this offset from the position.
    offset_x, offset_y = mask_centroid_offset
    center_x = int(round(position[0] - offset_x))
    center_y = int(round(position[1] - offset_y))

    y_start = max(0, center_y - template_h // 2)
    y_end = min(result.shape[0], y_start + template_h)
    x_start = max(0, center_x - template_w // 2)
    x_end = min(result.shape[1], x_start + template_w)

    template_y_start = max(0, template_h // 2 - (center_y - y_start))
    template_y_end = template_y_start + (y_end - y_start)
    template_x_start = max(0, template_w // 2 - (center_x - x_start))
    template_x_end = template_x_start + (x_end - x_start)

    # Check if cell is completely outside frame or slicing results in empty regions
    if y_start >= y_end or x_start >= x_end:
        return result.astype(np.uint8)
    
    if template_y_start >= template_y_end or template_x_start >= template_x_end:
        return result.astype(np.uint8)

    # Alpha blending for smooth anti-aliased compositing
    alpha = cell_mask[template_y_start:template_y_end, template_x_start:template_x_end]
    cell_slice = cell_image[template_y_start:template_y_end, template_x_start:template_x_end]
    target_region = result[y_start:y_end, x_start:x_end]
    binary_slice = binary_mask[template_y_start:template_y_end, template_x_start:template_x_end]
    
    # Verify shapes match before blending (safety check for edge cases)
    if alpha.shape != cell_slice.shape or alpha.shape != target_region.shape:
        return result.astype(np.uint8)
    
    # Alpha blending formula: result = alpha * foreground + (1 - alpha) * background
    blended = alpha * cell_slice + (1.0 - alpha) * target_region
    result[y_start:y_end, x_start:x_end] = blended

    # Store area if requested
    if area_store is not None and frame_idx is not None:
        area_store[frame_idx] = int(np.count_nonzero(binary_slice))
    
    return result.astype(np.uint8)
