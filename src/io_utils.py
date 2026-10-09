"""Dataset loading helpers for DRIVE / STARE / CHASE_DB1 fundus images.

Only DRIVE ships a field-of-view (FOV) mask per image. STARE and
CHASE_DB1 do not — callers should pass ``fov_mask_path=None`` for those
and downstream stage-1 code will simply skip the masking step.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_fundus_image(path: str) -> np.ndarray:
    """Load a fundus photo as RGB uint8, shape (H, W, 3).

    OpenCV reads images as BGR by default, so we convert to RGB to keep
    channel indexing intuitive (e.g. ``rgb[:, :, 1]`` really is green).
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_fov_mask(path: str | None) -> np.ndarray | None:
    """Load a binary field-of-view mask (uint8, values 0/255).

    Parameters
    ----------
    path:
        Path to the mask file (DRIVE provides one per image). Pass
        ``None`` for datasets that don't ship a mask (STARE, CHASE_DB1)
        — the caller then skips masking entirely.
    """
    if path is None:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read FOV mask at: {path}")
    return mask
