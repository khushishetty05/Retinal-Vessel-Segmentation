"""Stage 1 — Preprocessing & Contrast Enhancement.

Pipeline: green-channel extraction -> CLAHE -> morphological top-hat.

This module intentionally contains only the reusable logic (no
plotting, no notebook-only code) so Stage 2 (Frangi / matched-filter /
Gabor) and any later stage can import ``run_stage1`` as a plain
function, and the interactive notebook can call the same functions
without duplicating the algorithm.
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_green_channel(rgb: np.ndarray) -> np.ndarray:
    """Isolate the green channel of an RGB fundus image.

    Hemoglobin absorbs green light most strongly, so vessels are
    darkest (highest contrast against tissue) in this band -- plain
    grayscale conversion would blend in the oversaturated red channel
    and the poorly-illuminated blue channel, both of which reduce
    vessel contrast.

    Parameters
    ----------
    rgb: (H, W, 3) uint8 array, RGB channel order.

    Returns
    -------
    (H, W) uint8 array.
    """
    return rgb[:, :, 1]


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Normalizes the uneven illumination across the retina by
    equalizing local tiles, with ``clip_limit`` capping how much
    contrast can be amplified per tile so flat background regions
    don't get amplified into false micro-vessels.

    Parameters
    ----------
    gray: (H, W) uint8 array (typically the green channel).
    clip_limit: contrast amplification cap. Lower = safer against
        noise, higher = more local contrast.
    tile_grid_size: number of tiles (rows, cols) the image is divided
        into. Should be scaled up for larger images (e.g. CHASE_DB1's
        999x960) relative to smaller ones (DRIVE's 565x584) so tile
        size in pixels stays roughly consistent.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def tophat_filter(img: np.ndarray, disk_radius: int = 9) -> np.ndarray:
    """Morphological black top-hat: T_black(I) = closing(I, S) - I.

    Retinal vessels are DARK relative to their local background in the
    green/CLAHE image (hemoglobin absorbs green light), so extracting
    them requires the BLACK top-hat, not the white one: closing(I)
    fills in the thin dark vessel lines up to the surrounding
    background level, and subtracting the original I from that leaves
    a large positive value exactly where a dark ridge was erased, and
    ~0 everywhere the image already equaled its own closing (flat
    background, or features already brighter than their surroundings
    like the optic disc). The design formula in the source material is
    written as the white top-hat T_white(I) = I - opening(I, S), which
    is the correct-looking formula for the WRONG polarity here -- using
    it directly on a non-inverted green channel image suppresses
    vessels to zero instead of isolating them, since it targets small
    BRIGHT features, not the dark ones we actually have. cv2.MORPH_BLACKHAT
    is the mirror operation for dark features and is what's used below,
    so the module's actual output matches its documented contract
    (vessel ridges bright, background flat).

    Uses a disk-shaped structuring element (orientation-agnostic, since
    vessels run in every direction across the image) sized larger than
    the widest expected vessel, so the closing step fully fills even
    the thickest vessel trunks while leaving broad background
    structures (e.g. the optic disc) intact -- the subtraction then
    isolates the thin vessel ridges and flattens everything else.

    Parameters
    ----------
    img: (H, W) uint8 array (typically the post-CLAHE image).
    disk_radius: structuring element radius in pixels. Must exceed the
        widest vessel in the dataset (~1-15 px in DRIVE/STARE/CHASE_DB1);
        default of 9 (19x19 disk) safely clears that range. Increase if
        the optic disc still bleeds through; decrease if thin vessels
        start disappearing.
    """
    ksize = 2 * disk_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def run_stage1(
    rgb: np.ndarray,
    fov_mask: np.ndarray | None = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    disk_radius: int = 9,
) -> np.ndarray:
    """Run the full Stage 1 pipeline: green channel -> CLAHE -> top-hat.

    If ``fov_mask`` is provided, it's applied *after* CLAHE/top-hat so
    the local-neighborhood math still sees real image content at the
    field-of-view boundary rather than a synthetic black surround --
    only the final result is zeroed outside the mask, to avoid a bright
    artifact ring at the FOV edge.

    Parameters
    ----------
    rgb: (H, W, 3) uint8 raw fundus image, RGB order.
    fov_mask: optional (H, W) uint8 mask (0/255). Pass None for
        datasets without a shipped mask (STARE, CHASE_DB1).

    Returns
    -------
    (H, W) uint8 array: illumination-normalized image with vessel
    ridges bright and background flat, ready for Stage 2's directional
    filters (Frangi / matched-filter / Gabor).
    """
    green = extract_green_channel(rgb)
    enhanced = apply_clahe(green, clip_limit, tile_grid_size)
    tophat = tophat_filter(enhanced, disk_radius)
    if fov_mask is not None:
        tophat = np.where(fov_mask > 0, tophat, 0).astype(np.uint8)
    return tophat
