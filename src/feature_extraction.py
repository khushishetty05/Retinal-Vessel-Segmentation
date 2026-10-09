"""Stage 2 — Multi-Scale Directional Feature Extraction.

Runs three complementary directional filters on Stage 1's output
(green channel -> CLAHE -> top-hat, from ``src.preprocessing``):

  2a. Frangi Hessian vesselness   (skimage.filters.frangi)
  2b. Matched filter bank         (custom Gaussian/Cauchy kernels, no
                                    library equivalent exists)
  2c. Gabor wavelet bank          (skimage.filters.gabor)

Each produces its own (H, W) response map. Fusing them into one
composite feature map is deliberately NOT done here -- that's a
separate decision left to a later stage, once each map's behavior has
been inspected on its own.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import frangi as _skimage_frangi
from skimage.filters import gabor as _skimage_gabor


def frangi_vesselness(
    img: np.ndarray,
    sigmas: range = range(1, 5, 1),
    beta: float = 0.5,
    gamma: float | None = None,
) -> np.ndarray:
    """Frangi Hessian vesselness measure.

    Scores each pixel by how "tube-like" its local neighborhood is,
    via the eigenvalues of the Hessian matrix at each scale in
    ``sigmas``: R_B = |lambda1|/|lambda2| (tube vs. blob) and
    S = sqrt(lambda1^2 + lambda2^2) (structureness vs. flat
    background), combined into two exponential terms.

    Parameters
    ----------
    img: (H, W) array, Stage 1 output -- vessels must be BRIGHT ridges
        on a dark background (that's what top-hat produces).
    sigmas: scale-space range, roughly matching vessel half-width in
        pixels. Default 1-4 covers thin/medium capillaries; very thick
        trunks are left to the matched-filter/Gabor banks instead of
        widening this further (which starts flagging the optic-disc
        margin as a "wide tube").
    beta: sensitivity to blob-like (non-tube) structures.
    gamma: sensitivity to structureness/background. None lets skimage
        auto-derive it as half the max Hessian norm in this image,
        which adapts across datasets with different intensity ranges.

    Returns
    -------
    (H, W) float64 array in [0, 1]. Higher = more vessel-like.
    """
    img_f = img.astype(np.float64)
    return _skimage_frangi(
        img_f, sigmas=sigmas, beta=beta, gamma=gamma, black_ridges=False
    )


def _matched_filter_kernel(
    theta_deg: float,
    sigma: float = 2.0,
    length: int = 9,
    kernel_type: str = "gaussian",
) -> np.ndarray:
    """Build one matched-filter kernel oriented at ``theta_deg``.

    Evaluated directly from the cross-sectional-profile formula at
    each pixel's ROTATED coordinates (not by rasterizing a theta=0
    kernel and warping the image of it), so there's no interpolation
    blur distorting the exact profile:

        gaussian: K(x, y) = exp(-x^2 / (2*sigma^2))
        cauchy:   K(x, y) = 1 / (1 + (x/sigma)^2)   [heavier-tailed;
            our own literature-consistent approximation of the design's
            "Cauchy distribution kernel" -- the exact published Cauchy
            matched-filter formula wasn't in the source material we
            have, so this shape choice is a documented best-effort
            stand-in, not a verbatim transcription]

    valid for |y| <= length/2, |x| <= 3*sigma (the vessel cross-section
    window); zero everywhere else.

    NOTE on sign: the design source writes this with a leading minus
    (K = -exp(...)), which is the classic Chaudhuri et al. (1989) form
    for correlating against a RAW fundus image where vessels are DARK
    valleys. This pipeline's Stage 1 output has the opposite,
    deliberately-chosen polarity -- vessels are BRIGHT ridges (via the
    black top-hat, same convention Frangi's black_ridges=False already
    relies on) -- so the kernel here is a positive bump, matching that
    profile. Using the literal negative-sign formula against a
    bright-ridge image would correlate the *inverse* shape and produce
    a strongly negative response exactly at real vessels (confirmed
    empirically while validating this module); this positive-bump
    version is the sign-correct match for this pipeline's own
    Stage-1 convention, not the raw-image convention.

    The kernel is mean-subtracted over its own support only (not over
    the padded zero region around it), so flat/uniform image regions
    produce ~0 response instead of the filter's raw (nonzero-DC)
    output.
    """
    x_extent = int(np.ceil(3 * sigma))
    half_len = length / 2.0
    box_half = int(np.ceil(max(x_extent, half_len) * 1.5)) + 1
    coords = np.arange(-box_half, box_half + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")

    theta = np.deg2rad(theta_deg)
    # Map each pixel's (image-frame) coordinates back into the
    # kernel's own frame, where x' is across the vessel and y' runs
    # along it -- this is the inverse rotation by theta.
    x_prime = xx * np.cos(theta) + yy * np.sin(theta)
    y_prime = -xx * np.sin(theta) + yy * np.cos(theta)

    support = (np.abs(x_prime) <= x_extent) & (np.abs(y_prime) <= half_len)

    if kernel_type == "gaussian":
        values = np.exp(-(x_prime**2) / (2 * sigma**2))
    elif kernel_type == "cauchy":
        values = 1.0 / (1.0 + (x_prime / sigma) ** 2)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}")

    kernel = np.zeros_like(values)
    kernel[support] = values[support]
    kernel[support] -= kernel[support].mean()
    return kernel.astype(np.float32)


def matched_filter_bank(
    img: np.ndarray,
    sigma: float = 2.0,
    length: int = 9,
    n_orientations: int = 12,
    kernel_type: str = "gaussian",
) -> np.ndarray:
    """Matched filter bank: rotate the kernel across orientations, take
    the per-pixel MAX response.

    Only 0-180 degrees is covered (not 0-360) -- a line/tube detector
    at theta and theta+180 responds identically, so n_orientations=12
    at 15-degree steps (the design's stated increment) already covers
    every distinct orientation.

    Parameters
    ----------
    img: (H, W) array, Stage 1 output.
    sigma: cross-section width (pixels), matching thin/medium vessels.
    length: kernel length along the vessel axis (the L in |y|<=L/2).
    n_orientations: number of rotation steps across 0-180 degrees.
    kernel_type: "gaussian" or "cauchy" (see `_matched_filter_kernel`).

    Returns
    -------
    (H, W) float32 array (unbounded scale -- normalize before comparing
    to other maps).
    """
    img_f = img.astype(np.float32)
    step_deg = 180.0 / n_orientations
    responses = []
    for i in range(n_orientations):
        kernel = _matched_filter_kernel(
            i * step_deg, sigma=sigma, length=length, kernel_type=kernel_type
        )
        responses.append(cv2.filter2D(img_f, -1, kernel, borderType=cv2.BORDER_REPLICATE))
    return np.maximum.reduce(responses)


def gabor_filter_bank(
    img: np.ndarray,
    frequency: float = 0.15,
    n_orientations: int = 12,
) -> np.ndarray:
    """Gabor wavelet bank: rotate through orientations, take the
    per-pixel MAX of the response magnitude.

    Uses skimage.filters.gabor (frequency/bandwidth parameterization,
    correctly-sized envelope derived internally) rather than
    cv2.getGaborKernel, which would require manually picking a kernel
    pixel size and risks truncating the Gaussian envelope.

    The magnitude sqrt(real^2 + imag^2) of the complex response is
    used (not just the real part) so the result is phase-invariant --
    real and imaginary parts are 90-degrees out of phase, so either
    one alone can be near zero at a vessel purely due to its
    fractional offset within the wave's period, not its actual
    strength.

    Parameters
    ----------
    img: (H, W) array, Stage 1 output.
    frequency: cycles/pixel; roughly the inverse of expected vessel
        width so a full oscillation period spans a couple of vessel
        widths (default 0.15 -> ~6.7 px, thin/medium capillary range).
    n_orientations: number of rotation steps across 0-180 degrees.

    Returns
    -------
    (H, W) float64 array (unbounded scale -- normalize before
    comparing to other maps).
    """
    img_f = img.astype(np.float64)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0
    responses = []
    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        real, imag = _skimage_gabor(img_f, frequency=frequency, theta=theta)
        responses.append(np.sqrt(real**2 + imag**2))
    return np.maximum.reduce(responses)


def run_stage2(
    stage1_img: np.ndarray,
    frangi_kwargs: dict | None = None,
    matched_kwargs: dict | None = None,
    gabor_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    """Run all Stage 2 filters independently on Stage 1's output.

    Deliberately returns each response map UNFUSED -- combining them
    (normalization + weighting/combination strategy) is a distinct
    decision best made after inspecting each map's real behavior, and
    is left to a later stage rather than baked in here.

    Parameters
    ----------
    stage1_img: (H, W) array, the output of ``src.preprocessing.run_stage1``.
    frangi_kwargs, matched_kwargs, gabor_kwargs: optional per-filter
        overrides forwarded to `frangi_vesselness`, `matched_filter_bank`
        (used for both kernel types), and `gabor_filter_bank`.

    Returns
    -------
    dict with keys "frangi", "matched_gaussian", "matched_cauchy", "gabor",
    each an (H, W) array on its own, unnormalized scale.
    """
    frangi_kwargs = frangi_kwargs or {}
    matched_kwargs = matched_kwargs or {}
    gabor_kwargs = gabor_kwargs or {}

    return {
        "frangi": frangi_vesselness(stage1_img, **frangi_kwargs),
        "matched_gaussian": matched_filter_bank(
            stage1_img, kernel_type="gaussian", **matched_kwargs
        ),
        "matched_cauchy": matched_filter_bank(
            stage1_img, kernel_type="cauchy", **matched_kwargs
        ),
        "gabor": gabor_filter_bank(stage1_img, **gabor_kwargs),
    }
