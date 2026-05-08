# Barnacle Counter — Automated CV Pipeline

## Project Overview

National Park Service scientists place a fixed-size teal/dark-green wire frame over barnacle colonies in coastal tide pools, photograph the frame, then manually count the barnacles inside it. With counts often exceeding 1,000 barnacles per image, manual counting is a significant bottleneck.

This project builds a two-stage Python/OpenCV pipeline to automate that count:

- **Stage 1** — Detect the grid frame and crop to the inner region of interest (ROI).
- **Stage 2** — Detect barnacle circles/ellipses within the cropped ROI and return an estimated count.

---

## Repository Structure

```
barnacle-counter/
├── src/
│   ├── crop.py              # Stage 1: grid detection → perspective-corrected crop
│   ├── tuning.py            # Stage 1 parameter sweep (outputs annotated images)
│   ├── barnacle_counter.py  # Stage 2: barnacle detection and counting
│   ├── barnacle_tuner.py    # Stage 2 parameter sweep (grid search over Hough params)
│   ├── ellipse_tuner.py     # Stage 2 parameter sweep focused on ellipse detection
│   └── contour_tuner.py     # Stage 2 parameter sweep using raw contour detection
├── media/
│   ├── img1.png, img2.png   # Raw input images
│   ├── output.jpg           # crop.py output
│   ├── final.jpg            # Annotated barnacle detection output
│   ├── output_tuning/       # Stage 1 tuning annotated images
│   ├── barnacle_runs/       # Stage 2 Hough tuning annotated images, CSV, and contact sheet
│   ├── ellipse_runs/        # Stage 2 ellipse tuning annotated images, CSV, and contact sheet
│   └── contour_runs/        # Stage 2 contour tuning annotated images, CSV, and contact sheet
└── README.md
```

---

## Stage 1 — ROI Extraction (`src/crop.py`)

### Goal
Detect the teal grid frame in the raw field image, isolate the interior square, and output a perspective-corrected crop. This removes background clutter and normalises the image for the downstream detector.

### Pipeline (in order)

1. **Load** image from `INPUT_PATH`.
2. **HSV mask** — convert to HSV and threshold to isolate the teal grid.
   - `LOWER_GREEN = [75, 40, 20]`
   - `UPPER_GREEN = [95, 255, 120]`
3. **Morphological cleanup** — dilate then erode to remove speckle noise while preserving thick grid lines.
4. **Canny edge detection** on the cleaned mask.
5. **HoughLinesP** to detect line segments.
   - Tunable via `HOUGH_THRESHOLD`, `MIN_LINE_LENGTH`, `MAX_LINE_GAP`.
6. **`separate_lines`** — splits segments into horizontals and verticals using a 45° threshold on `arctan2`.
7. **`grid_bounds`** — finds the crop rectangle:
   - Compute centroid of all detected segments.
   - Keep only horizontals whose x-range spans `cx`, and verticals whose y-range spans `cy` (eliminates outer frame fragments and noise).
   - Walk outward from the centroid: the closest spanning horizontal above/below → top/bottom boundary; closest spanning vertical left/right → left/right boundary.
8. **`warp_roi`** — corrects camera angle via `cv2.getPerspectiveTransform` + `cv2.warpPerspective` and writes the clean crop to `OUTPUT_PATH`.

### Key Design Decisions

| Decision | Rationale |
|---|---|
| 45° arctan2 split for H/V classification | Dominant-angle approach was mis-labelling segments; simple threshold is more robust |
| Centroid + axis-spanning filter | Ensures only lines that genuinely cross the interior are used as borders; excludes outer frame noise |
| Centroid-outward boundary selection | Finds the *innermost* bounding lines rather than the outermost detected extremes, matching the inner square the scientists want |

### Tuning Tool — `src/tuning.py`

Sweeps Hough parameters (3 thresholds × 3 min-lengths × 4 max-gaps = 36 combinations) and outputs annotated images with detected lines drawn on, saved to `media/output_tuning/`. Output filenames encode the parameter values (e.g. `thresh080_minLen400_maxGap200.png`) for easy visual comparison.

---

## Stage 2 — Barnacle Detection (`src/barnacle_counter.py`)

### Goal
Accept the cropped ROI image from Stage 1 and return an estimated barnacle count.

### Pipeline

1. **Preprocessing** — greyscale → bilateral filter (`d=9`, `sigmaColor=75`, `sigmaSpace=75`) → CLAHE (`clipLimit=2.5`, `tileGridSize=(8,8)`).
2. **Primary detector** — Hough Circle Transform (`cv2.HoughCircles` with `HOUGH_GRADIENT`).
3. **Optional fallback** — contour-based ellipse fitting via Otsu threshold → morphological close → `cv2.findContours` → `cv2.fitEllipse`. Filters by area and aspect ratio.
4. **Deduplication** — if ellipses are active, any circle whose centre is within the circle's own radius of an ellipse centre is dropped to avoid double-counting.
5. **Adjustment constant** — a user-defined multiplier applied to the raw detection count to compensate for systematic under- or over-detection.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `param1` | `50` | Canny upper threshold passed to Hough |
| `param2` | `35` | Accumulator threshold (lower = more detections) |
| `min_dist` | `25` | Minimum distance between detected circle centres (px) |
| `min_radius` / `max_radius` | `6` / `40` | Radius bounds for valid circles (px) |
| `dp` | `1` | Inverse resolution ratio of the accumulator |
| `use_ellipse` | `True` | Enable the contour-based ellipse fallback |
| `ellipse_min_area` / `ellipse_max_area` | `200` / `15000` | Area bounds for ellipse contour candidates (px²) |
| `ellipse_aspect_ratio` | `1.1` | Max major/minor axis ratio to accept as an ellipse |
| `adjustment` | `1.0` | Scalar multiplier on the raw count |

### Tuning Tool — `src/barnacle_tuner.py`

Performs a grid search over the parameters above (default: 4 `param2` × 3 `param1` × 3 `min_dist` × 3 `radius_pair` × 3 `dp` = 324 combinations). The ellipse fallback is **disabled** during the sweep (`use_ellipse=False`) to isolate Hough parameter effects.
- Saves annotated output images to `--out_dir` (default: `tuning_output/`) with parameter-tagged filenames (e.g. `img1__p1=80_p2=35_md=15_r=6-40_dp=1.0.jpg`).
- Writes `tuning_results.csv` with per-run counts and timing.
- Generates `contact_sheet.jpg` (thumbnails sorted by `raw_count` ascending) for visual side-by-side comparison.
- Runs with `raw_count >= max_count` are **skipped and not saved** (configurable via `--max_count`, default `200`).

### Ellipse-only Tuning Tool — `src/ellipse_tuner.py` *(experimental)*

Sweeps the full contour-based ellipse detection pipeline in isolation (no Hough circles), making it easier to find parameters that work well for non-circular barnacles.

**Parameter grid** (default: 3 × 3 × 3 × 3 × 3 = 243 combinations):

| Parameter | Values swept | Description |
|---|---|---|
| `thresh_method` | `otsu`, `adaptive`, `otsu_inv` | Binarisation strategy before contour finding |
| `morph_kernel` | `3`, `5`, `7` | Elliptical closing kernel size (px) — larger fills bigger rim gaps |
| `morph_iters` | `1`, `2`, `3` | Closing iterations — more iterations merge more gaps |
| `area_pair` | `(100,8000)`, `(200,15000)`, `(400,25000)` | Min/max contour area bounds (px²) |
| `max_aspect` | `1.5`, `2.0`, `3.0` | Max major/minor axis ratio — lower keeps shapes closer to circular |

Outputs go to `media/ellipse_runs/` by default:
- Annotated images with parameter-tagged filenames (e.g. `output__th=otsu_k=5_i=2_a=200-15000_ar=2.0.jpg`).
- `ellipse_tuning_results.csv` with per-run counts and timing.
- `ellipse_contact_sheet.jpg` sorted by count ascending.
- Runs with `raw_count >= max_count` are skipped (default `200`).

### Raw Contour Tuning Tool — `src/contour_tuner.py`

Sweeps the same preprocessing and filtering parameters as `ellipse_tuner.py`, but skips `fitEllipse` entirely — each accepted contour is drawn directly onto the image as-is. This means any shape (circular, elongated, irregular, partially occluded) is accepted as long as it passes the area and solidity filters, with no aspect-ratio assumption.

**Parameter grid** (default: 3 × 3 × 2 × 3 × 3 = 162 combinations):

| Parameter | Values swept | Description |
|---|---|---|
| `thresh_method` | `otsu`, `adaptive`, `otsu_inv` | Binarisation strategy |
| `morph_kernel` | `3`, `5`, `7` | Closing kernel size (px) |
| `morph_iters` | `1`, `2` | Closing iterations |
| `area_pair` | `(50,1000)`, `(100,2000)`, `(200,3000)` | Min/max contour area bounds (px²) |
| `min_solidity` | `0.5`, `0.65`, `0.8` | Min contour_area / convex_hull_area |

Outputs go to `media/contour_runs/` by default:
- Annotated images with contour outlines and centroid dots drawn in cyan.
- `contour_tuning_results.csv` with per-run counts and timing.
- `contour_contact_sheet.jpg` sorted by count ascending.

---

## Data

| File | Description |
|---|---|
| `media/img1.png`, `media/img2.png` | Raw field images with the teal wire frame visible |
| `media/output.jpg` | Perspective-corrected crop produced by `crop.py` |
| `media/final.jpg` | Annotated barnacle detection output from `barnacle_counter.py` |
| `media/output_tuning/` | Annotated images from the Stage 1 parameter sweep (`tuning.py`) |
| `media/barnacle_runs/` | Annotated images, `tuning_results.csv`, and `contact_sheet.jpg` from the Stage 2 sweep (`barnacle_tuner.py`) |

---

## Evaluation

No ground-truth annotation masks are currently included in the repository. Recommended metrics once annotations are available:

- **Count accuracy** — `|predicted_count − true_count| / true_count`
- **Precision / Recall** — match predicted circles to ground-truth contour centroids within a distance threshold
- **F1 score** — harmonic mean of precision and recall

With only two input images and no labels, evaluation is currently qualitative (visual inspection of annotated outputs). The tuning tools are designed to surface parameter settings worth validating if annotated data becomes available.

---

## Next Steps

- Collect more annotated images to improve tuning and evaluation reliability.
- Explore learned approaches (e.g. fine-tuned instance segmentation) if the Hough-based prototype plateaus.
- Integrate Stages 1 and 2 into a single end-to-end script with a simple output (count + annotated image).
- Consider a lightweight review UI so scientists can quickly confirm or adjust the automated count before recording results.