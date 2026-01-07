"""Shared helpers for generating pixmaps used on the editing canvas."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtGui import QImage, QPixmap

from ...assets import CellTemplate
from ...core.feather import FeatherParameters, signed_distance_alpha
from ...core.noise import add_quantization_noise


def template_pixmap_with_offset(
    template: CellTemplate,
    feather_params: Optional[FeatherParameters],
    noise_stddev: float | None = None,
) -> Tuple[QPixmap, Tuple[float, float]]:
    """Build a pixmap for a cell template plus the offset needed to center it on trajectories.
    
    Args:
        template: Loaded ``CellTemplate`` describing the grayscale image/mask.
        feather_params: Optional feathering parameters; when active we reuse the
            signed-distance alpha map so edit-mode overlays match preview renders.
    
    Returns:
        Tuple of ``(pixmap, (offset_x, offset_y))`` ready to assign to a
        ``QGraphicsPixmapItem`` via ``item.setOffset``.
    """
    image = template.image.astype(np.uint8)
    mask_bool = template.mask.astype(bool)

    if noise_stddev is not None and noise_stddev > 0 and mask_bool.any():
        noisy_full = add_quantization_noise(image, noise_stddev)
        image = image.copy()
        image[mask_bool] = noisy_full[mask_bool]
    
    if feather_params and feather_params.is_active():
        alpha = signed_distance_alpha(mask_bool, feather_params)
        mask_uint8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    else:
        mask_uint8 = mask_bool.astype(np.uint8) * 255
    
    height, width = image.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = image
    rgba[..., 1] = image
    rgba[..., 2] = image
    rgba[..., 3] = mask_uint8
    
    qimage = QImage(rgba.data, width, height, QImage.Format_RGBA8888)
    pixmap = QPixmap.fromImage(qimage.copy())
    
    offset_x, offset_y = template.mask_centroid_offset
    item_offset = (-pixmap.width() / 2.0 - offset_x, -pixmap.height() / 2.0 - offset_y)
    return pixmap, item_offset
