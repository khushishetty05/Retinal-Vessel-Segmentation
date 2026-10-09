"""Stage 3 — Classical Post-processing and Thresholding.

This module takes the multiple continuous feature maps produced by
Stage 2 (feature_extraction) and combines them into a single binary
mask representing the final vessel segmentation.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_map(img: np.ndarray) -> np.ndarray:
    """Min-max normalize a continuous feature map to [0, 1].

    Parameters
    ----------
    img: (H, W) continuous array (float or int).

    Returns
    -------
    (H, W) float64 array bounded strictly in [0.0, 1.0].
    """
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.float64)
    return (img - img_min) / (img_max - img_min)


def fuse_features(
    feature_dict: dict[str, np.ndarray], method: str = "mean"
) -> np.ndarray:
    """Fuse multiple normalized feature maps into a single continuous map.

    Parameters
    ----------
    feature_dict: dict mapping string names to (H, W) arrays.
    method: "mean" or "max". How to combine the maps.

    Returns
    -------
    (H, W) float64 array in [0.0, 1.0].
    """
    if not feature_dict:
        raise ValueError("feature_dict cannot be empty.")

    # Normalize all maps to [0, 1] first
    maps = [normalize_map(m) for m in feature_dict.values()]
    stack = np.stack(maps, axis=0)

    if method == "mean":
        fused = np.mean(stack, axis=0)
    elif method == "max":
        fused = np.max(stack, axis=0)
    else:
        raise ValueError(f"Unknown fusion method: {method!r}")
        
    # Re-normalize just to be safe, though mean/max of [0, 1] is in [0, 1]
    return normalize_map(fused)


def apply_threshold(
    fused_map: np.ndarray, method: str = "otsu", manual_thresh: float = 0.5
) -> np.ndarray:
    """Convert a continuous fused map [0, 1] to a binary vessel mask.

    Parameters
    ----------
    fused_map: (H, W) array in [0.0, 1.0].
    method: "otsu" for automatic global thresholding, or "manual".
    manual_thresh: threshold value if method="manual".

    Returns
    -------
    (H, W) uint8 array, where 255 is vessel and 0 is background.
    """
    # Convert [0, 1] float to [0, 255] uint8 for thresholding
    img_uint8 = (fused_map * 255).astype(np.uint8)

    if method == "otsu":
        # Otsu's method automatically calculates the optimal threshold
        _, binary = cv2.threshold(
            img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "manual":
        # Manual global threshold
        thresh_val = int(manual_thresh * 255)
        _, binary = cv2.threshold(img_uint8, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown threshold method: {method!r}")

    return binary


def run_stage3_classical(
    feature_dict: dict[str, np.ndarray],
    fov_mask: np.ndarray | None = None,
    fusion_method: str = "mean",
    threshold_method: str = "otsu",
    manual_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run all classical Stage 3 steps to produce a final binary mask.

    Parameters
    ----------
    feature_dict: the output of src.feature_extraction.run_stage2.
    fov_mask: optional (H, W) boolean/uint8 mask of the retina. If
        provided, any predicted vessels outside the retina are zeroed out.
    fusion_method: "mean" or "max".
    threshold_method: "otsu" or "manual".

    Returns
    -------
    fused_map: (H, W) float64 array [0, 1], the combined continuous map.
    binary_mask: (H, W) uint8 array (0 or 255), the final vessel mask.
    """
    fused_map = fuse_features(feature_dict, method=fusion_method)
    binary_mask = apply_threshold(
        fused_map, method=threshold_method, manual_thresh=manual_thresh
    )

    if fov_mask is not None:
        # Blank out anything outside the FOV mask
        binary_mask = np.where(fov_mask > 0, binary_mask, 0).astype(np.uint8)

    return fused_map, binary_mask
