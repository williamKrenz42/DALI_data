# Barnacle Counter — Automated CV Pipeline

## Project Overview

National Park Service scientists place a fixed-size teal/dark-green wire frame over barnacle colonies in coastal tide pools, photograph the frame, then manually count the barnacles inside it. With counts often exceeding 1,000 barnacles per image, manual counting is a significant bottleneck.

This project builds a two-stage Python/OpenCV pipeline to automate that count:

- **Stage 1** — Detect the grid frame and crop to the inner region of interest (ROI).
- **Stage 2** — Detect barnacle circles/ellipses within the cropped ROI and return an estimated count.

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

### Tuning Tool — `src/tuning.py`

Sweeps Hough parameters (3 thresholds × 3 min-lengths × 4 max-gaps = 36 combinations) and outputs annotated images with detected lines drawn on, saved to `media/output_tuning/`. Output filenames encode the parameter values (e.g. `thresh080_minLen400_maxGap200.png`) for easy visual comparison.

---

## Stage 2 — Barnacle Detection (`src/barnacle_counter.py`)

### Goal
Accept the cropped ROI image from Stage 1 and return an estimated barnacle count.

### Simple GUI

Run the counter without arguments to open a basic upload window:

```bash
python src/barnacle_counter.py
```

Choose a raw framed image. The app runs the Stage 1 crop, runs the barnacle detector on the cropped ROI, then displays the estimated count and annotated overlay. It also writes the latest crop to `media/output.jpg` and the latest overlay to `media/final.jpg`.

For scripted runs on an already-cropped image, keep using:

```bash
python src/barnacle_counter.py --image media/output.jpg --output media/final.jpg
```

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

## Evaluation

From the images that the model outputs, my model seems relatively accurate in its calculations. However, without a larger sample size of labled barnacle images, it is much harder to make a generalization. Additionally, the adjustment value I apply to each count was very difficult to judge with such a small sample size, given more time and data, I would have loved to arrive at a closer adjustment based purely in statistics.

---

## Next Steps

- Collect more annotated images to improve tuning and evaluation reliability.
- Explore learned approaches (e.g. fine-tuned instance segmentation) if the Hough-based prototype plateaus.
- Improve and include ellipse and countour detection models.
- Consider a lightweight review UI so scientists can quickly confirm or adjust the automated count before recording results.


---

## Learning Process

- The main parts of my learning process for this project was learning of, and experimenting with the different ways to use OpenCV to detect the barnacles. I was essentially new to OpenCV coming into this project, so I honeslty spent the majority of my time learning how the different parameters to each detection function interact to actually label the images. The first step was to do background research into each of the methods, circle, line, ellipse, and contour to discover how to actually use them, then I spent some time tuning the different approaches to many different combinations of parameters to see what would work best in these cases. This was very interesting, and I would love to do a longer project with a larger dataset of labled images to really tune in the models.
