# Retinal Vessel Segmentation (Classical / No Deep Learning)

This project extracts the blood-vessel network from color retinal photographs
("fundus images") using purely classical image-processing techniques — no
neural networks, no training, no GPU. It's built for a Digital Image
Processing course, so every step is a deterministic, inspectable math
operation rather than a black box.

Right now the project implements **Stage 1 (preprocessing)** and **Stage 2
(directional feature extraction)**. Later stages (thresholding into a final
black/white vessel mask, and cleanup) are not implemented yet.

---

## Part 1 — Setup, data, and how to run it

### 1. Install Python dependencies

You need Python 3.10+ (developed on 3.11). From the project root:

```bash
pip install -r requirements.txt
```

This installs:
- `numpy` — array math
- `opencv-python` (imported as `cv2`) — image I/O, CLAHE, morphology, convolution
- `scikit-image` (imported as `skimage`) — Frangi vesselness filter, Gabor filter
- `matplotlib` — displaying/saving images
- `jupyter`, `notebook`, `ipywidgets` — running the interactive notebooks below

### 2. Download the datasets

The code is written against three standard public retinal-image datasets.
You only need one to get started (**DRIVE** is recommended first — it's the
smallest and cleanest):

| Dataset | Where to get it | Images | Has a mask file? |
|---|---|---|---|
| **DRIVE** | [drive.grand-challenge.org](https://drive.grand-challenge.org/) (free registration required) | 40 (565×584) | Yes |
| **STARE** | [cecas.clemson.edu/~ahoover/stare](http://cecas.clemson.edu/~ahoover/stare/) | 20 (700×605) | No |
| **CHASE_DB1** | [blogs.kingston.ac.uk/retinal/chasedb1](https://blogs.kingston.ac.uk/retinal/chasedb1/) | 28 (999×960) | No |

"Has a mask file" means: DRIVE ships a black-and-white "field of view" (FOV)
mask per image marking which pixels are actually inside the circular retina
photo (vs. the black surround) — this is used to blank out that surrounding
border. STARE and CHASE_DB1 don't provide this, and the code handles that
(you just pass `None` for the mask).

### 3. Where to put the downloaded images

Create this folder structure under `data/` (this folder is git-ignored, so
your downloaded images never get committed):

```
data/
└── drive/
    ├── images/
    │   └── 21_training.tif        <- one raw color fundus photo
    └── mask/
        └── 21_training_mask.gif   <- its matching FOV mask
```

For STARE/CHASE_DB1, use the same pattern minus the `mask/` folder, e.g.
`data/stare/images/im0001.ppm`.

The exact file names don't matter — you just need to point the notebooks at
whichever file you downloaded (next step).

### 4. Run it

```bash
jupyter notebook
```

This opens Jupyter in your browser. Then, in order:

1. Open **`notebooks/01_stage1_preprocessing.ipynb`**.
   - In the second code cell, edit `IMAGE_PATH` and `MASK_PATH` to point at
     the actual file you downloaded (e.g.
     `data/drive/images/21_training.tif`). Set `MASK_PATH = None` if your
     dataset has no mask file (STARE/CHASE_DB1).
   - Run all cells top to bottom (`Run` → `Run All Cells`). You'll see a
     4-panel figure: the original photo, its green channel, the
     contrast-enhanced version, and the final vessel-highlighted output.
2. Open **`notebooks/02_stage2_feature_extraction.ipynb`** and do the same
   with the same `IMAGE_PATH`/`MASK_PATH`. This runs three different vessel
   detectors on Stage 1's output and shows all of them side by side.

Every run also saves its comparison figure as a PNG under `outputs/stage1/`
or `outputs/stage2/`, so you can look back at results without re-running
anything.

---

## Part 2 — How the pipeline works, file by file

### The big picture

```
raw color photo (data/drive/images/*.tif)
        │
        ▼
  src/io_utils.py        →  loads the image (and its mask, if any) into memory
        │
        ▼
  src/preprocessing.py   →  STAGE 1: cleans up lighting, makes vessels stand out
        │                   (output: one grayscale image, vessels bright)
        ▼
  src/feature_extraction.py → STAGE 2: runs 3 different "does this look like
        │                      a blood vessel?" detectors on Stage 1's output
        ▼
  (not built yet) STAGE 3 → would turn the detector outputs into a final
                             black-and-white vessel map
```

The two notebooks in `notebooks/` don't contain any of the real logic — they
just import the functions below, run them on one image, and display the
results. All the actual image-processing code lives in `src/`.

Each function below is shown with its actual signature (and, where it's
short enough to matter, its actual body) so you can match it up 1:1 with
what's in the file rather than hunting for it by name alone.

### `src/io_utils.py` — loading images

```python
def load_fundus_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```
**`load_fundus_image(path)`** — reads a color photo file into a plain numpy
array of pixel values (red, green, blue per pixel), shape `(H, W, 3)`.
Handles the detail that OpenCV normally reads colors in "BGR" order
internally, and converts it to the more intuitive "RGB" order (so
`rgb[:, :, 1]` elsewhere in the code really is the green channel).

```python
def load_fov_mask(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read FOV mask at: {path}")
    return mask
```
**`load_fov_mask(path)`** — reads the black/white mask file that marks
which part of the image is actual retina vs. black background border.
Pass `path=None` (because your dataset doesn't have one, e.g. STARE or
CHASE_DB1) and it just returns `None` — every later step checks for that
and skips masking entirely rather than erroring.

### `src/preprocessing.py` — Stage 1: cleaning up the photo

The problem this stage solves: raw fundus photos have uneven lighting (bright
in the middle, darker at the edges from the camera flash), and the blood
vessels are only faintly darker than the surrounding tissue. This stage fixes
both problems and produces one clean grayscale image where vessels are
clearly the brightest thing in the picture.

```python
def extract_green_channel(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 1]
```
**`extract_green_channel(rgb)`** — throws away the red and blue color data
and keeps only the green (index `1` of the last axis). Blood absorbs green
light the most, so vessels show up with the most contrast in this one color
channel — using the red or blue channel (or a plain grayscale average of
all three) would make the vessels harder to see, not easier.

```python
def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)
```
**`apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))`** —
"Contrast Limited Adaptive Histogram Equalization." Splits the image into
an 8×8 grid of tiles and boosts the contrast within each tile individually
(`cv2.createCLAHE`/`.apply()` do the actual histogram math), so the whole
retina ends up evenly lit instead of bright-in-the-middle/dark-at-the-edges.
`clip_limit` caps how much any one tile can be boosted, so it doesn't turn
random background noise into fake-looking vessels.

```python
def tophat_filter(img: np.ndarray, disk_radius: int = 9) -> np.ndarray:
    ksize = 2 * disk_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```
**`tophat_filter(img, disk_radius=9)`** — the key trick that actually
isolates the vessels. `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...)`
builds a disk-shaped "probe" (19 pixels across by default); `MORPH_BLACKHAT`
then compares the image to a version of itself with all thin dark lines
painted over ("closed" — filled in with their surroundings), and keeps only
the *difference* — i.e. exactly the small, thin, darker-than-their-
surroundings shapes, which is what blood vessels are, geometrically. Big
features like the optic disc (a large bright circular structure) are too
big for the disk probe to erase, so they disappear in the subtraction. The
output has vessels as bright lines on an otherwise near-black background.

```python
def run_stage1(
    rgb: np.ndarray,
    fov_mask: np.ndarray | None = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    disk_radius: int = 9,
) -> np.ndarray:
    green = extract_green_channel(rgb)
    enhanced = apply_clahe(green, clip_limit, tile_grid_size)
    tophat = tophat_filter(enhanced, disk_radius)
    if fov_mask is not None:
        tophat = np.where(fov_mask > 0, tophat, 0).astype(np.uint8)
    return tophat
```
**`run_stage1(rgb, fov_mask, ...)`** — runs the three functions above in
order (green channel → CLAHE → top-hat), and if a mask was provided, blacks
out everything outside the retina's circular boundary at the very end
(`np.where(fov_mask > 0, tophat, 0)`). This is the one function everything
else calls to get Stage 1's final result — it's what both notebooks and
Stage 2 actually import and run.

### `src/feature_extraction.py` — Stage 2: three vessel detectors

The problem this stage solves: Stage 1's output still has some noise and
disease spots (bleeding, lesions) that can look vessel-like. This stage runs
three independent, mathematically different tests for "is this pixel part of
a thin, tube-shaped structure?" — each one has different strengths and
blind spots, so running all three (rather than just one) catches more real
vessels while rejecting more false positives.

```python
def frangi_vesselness(
    img: np.ndarray,
    sigmas: range = range(1, 5, 1),
    beta: float = 0.5,
    gamma: float | None = None,
) -> np.ndarray:
    img_f = img.astype(np.float64)
    return _skimage_frangi(
        img_f, sigmas=sigmas, beta=beta, gamma=gamma, black_ridges=False
    )
```
**`frangi_vesselness(img, ...)`** — for every pixel, mathematically checks
the *shape* of the local intensity pattern: is it shaped like a tube (a real
vessel), a round blob (a lesion or noise spot), or basically flat background?
It scores each pixel by how "tube-like" it is, returning values from 0 to 1.
This comes from a ready-made, well-tested library function
(`skimage.filters.frangi`, imported at the top of the file as
`_skimage_frangi`) rather than being written from scratch, since the
underlying math (Hessian-matrix eigenvalues) is easy to get subtly wrong by
hand. `black_ridges=False` tells it vessels are the *bright* things here
(matching what `tophat_filter` produces), not the dark things a raw photo
would have.

```python
def _matched_filter_kernel(
    theta_deg: float,
    sigma: float = 2.0,
    length: int = 9,
    kernel_type: str = "gaussian",
) -> np.ndarray:
    ...
    if kernel_type == "gaussian":
        values = np.exp(-(x_prime**2) / (2 * sigma**2))
    elif kernel_type == "cauchy":
        values = 1.0 / (1.0 + (x_prime / sigma) ** 2)
    ...
    kernel[support] -= kernel[support].mean()
    return kernel.astype(np.float32)


def matched_filter_bank(
    img: np.ndarray,
    sigma: float = 2.0,
    length: int = 9,
    n_orientations: int = 12,
    kernel_type: str = "gaussian",
) -> np.ndarray:
    ...
    for i in range(n_orientations):
        kernel = _matched_filter_kernel(i * step_deg, sigma, length, kernel_type)
        responses.append(cv2.filter2D(img_f, -1, kernel, ...))
    return np.maximum.reduce(responses)
```
**`matched_filter_bank(img, ...)`** (using the helper
**`_matched_filter_kernel(...)`**) — builds a small mathematical "stencil"
shaped exactly like a cross-section of a blood vessel (a bump shape —
`exp(...)` for a smooth Gaussian bump, or `1/(1+...)` for a "Cauchy" bump
with heavier tails that's a bit more forgiving of real vessels' slightly
irregular edges), then slides it over the image (`cv2.filter2D`) at 12
different rotation angles in a loop (since vessels can point in any
direction) and keeps, for each pixel, the strongest match found at any
angle (`np.maximum.reduce`). This one is custom-built in this project
because no standard library ships a retina-specific version of it.

```python
def gabor_filter_bank(
    img: np.ndarray,
    frequency: float = 0.15,
    n_orientations: int = 12,
) -> np.ndarray:
    ...
    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        real, imag = _skimage_gabor(img_f, frequency=frequency, theta=theta)
        responses.append(np.sqrt(real**2 + imag**2))
    return np.maximum.reduce(responses)
```
**`gabor_filter_bank(img, ...)`** — same rotate-and-keep-the-best idea as
the matched filter, but uses a different, ready-made filter shape from
`skimage.filters.gabor` (imported as `_skimage_gabor`) — a "Gabor wavelet,"
a ripple pattern inside a soft envelope — that's particularly good at
picking up the thinnest, faintest capillaries the other two methods might
miss. `np.sqrt(real**2 + imag**2)` combines the filter's two output parts
into one strength value per pixel.

```python
def run_stage2(
    stage1_img: np.ndarray,
    frangi_kwargs: dict | None = None,
    matched_kwargs: dict | None = None,
    gabor_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    ...
    return {
        "frangi": frangi_vesselness(stage1_img, **frangi_kwargs),
        "matched_gaussian": matched_filter_bank(stage1_img, kernel_type="gaussian", **matched_kwargs),
        "matched_cauchy": matched_filter_bank(stage1_img, kernel_type="cauchy", **matched_kwargs),
        "gabor": gabor_filter_bank(stage1_img, **gabor_kwargs),
    }
```
**`run_stage2(stage1_img, ...)`** — runs all three detectors above on Stage
1's output and hands back their four results (two matched-filter variants
count separately) as a dictionary, without combining them into one final
answer yet. Keeping them separate at this point lets you look at each
detector's result on its own before deciding how to combine them — that
combination step is the next piece of this project, not yet built.

### What's next (not built yet)

Stage 3 would take the three maps from `run_stage2` and combine them into
one map, then turn that into a final black-and-white "vessel / not vessel"
decision per pixel (classically, via automatic thresholding — no neural
network). Stage 4 would then clean up that black-and-white result by
removing small leftover noise blobs and thinning the vessels down to
single-pixel-wide lines.
