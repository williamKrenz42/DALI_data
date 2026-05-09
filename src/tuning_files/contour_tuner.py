"""
contour_tuner.py
-----------------
Grid-searches contour-based barnacle rim detection parameters, saves one
annotated image per combination with a tagged filename, and writes a CSV +
contact sheet so you can compare results at a glance.

Target structure: the dark annular rim visible on top of each live barnacle —
a ring shape, not a filled blob. The pipeline is tuned for this specifically:

  preprocess → threshold → morphological OPEN (denoise without filling the ring)
  → findContours with RETR_CCOMP (captures ring topology via parent/child hierarchy)
  → keep only contours that have a child hole (i.e. are ring-shaped)
  → filter by rim area and annularity (area / bounding-circle area)

Usage:
    python src/contour_tuner.py --image media/output.jpg
    python src/contour_tuner.py --image media/output.jpg --out_dir media/contour_runs/

Customise the PARAM_GRID dict below to change what gets swept.

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace


# ══════════════════════════════════════════════════════════════════════════════
#  PARAM GRID  –  edit this to control what gets swept
#  Each key maps to a list of values to try.
#  Total runs = product of all list lengths.
# ══════════════════════════════════════════════════════════════════════════════
PARAM_GRID = {
    # ── Binarisation method ───────────────────────────────────────────────────
    # "otsu"     : single global threshold, good for even lighting
    # "otsu_inv" : same but inverted, for light barnacles on dark substrate
    # "adaptive"     : per-neighbourhood threshold, best for uneven lighting
    # "adaptive_inv" : same but inverted, for dark rims on pale shell plates
    #
    # The thin top rim in close-up barnacle images is darker than the shell
    # surface, so the sweep now prioritises inverted thresholding.
    "thresh_method": ["otsu_inv", "adaptive_inv"],

    # ── Morphological opening kernel size (px) ────────────────────────────────
    # OPEN = erode then dilate: removes speckle noise while leaving ring
    # structures intact. Unlike closing, it does NOT fill the hole in the ring.
    # Larger kernel removes bigger noise but can break thin rims.
    "morph_kernel": [1, 3],

    # ── Morphological opening iterations ─────────────────────────────────────
    "morph_iters": [1],

    # ── Rim contour area bounds (px²) — as (min, max) pairs ──────────────────
    # This is the area of the rim band itself (not the whole barnacle disc).
    # Expect it to be smaller than in the blob-based tuners.
    "area_pair": [(20, 300), (40, 800), (75, 1500), (150, 3000)],

    # ── Annularity bounds — as (min, max) pairs ───────────────────────────────
    # annularity = contour_area / bounding_circle_area
    # A perfect thin ring scores low (small band, large bounding circle).
    # A filled disc scores close to 1.0.
    # The target rim is a narrow dark band, so bias lower than the previous
    # broad-rim range and avoid accepting filled dark patches.
    "annularity_pair": [(0.03, 0.25), (0.05, 0.35), (0.08, 0.45)],
}

# Fixed params not being swept
FIXED = dict(
    adjustment = 1.0,
    debug      = False,
)

# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    thresh_method:   str
    morph_kernel:    int
    morph_iters:     int
    min_area:        int
    max_area:        int
    min_annularity:  float
    max_annularity:  float
    contours:        int
    adjusted:        float
    out_file:        str
    elapsed_ms:      float


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(image: np.ndarray) -> np.ndarray:
    """Greyscale → bilateral filter → CLAHE."""
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(blurred)


# ──────────────────────────────────────────────────────────────────────────────
# Thresholding
# ──────────────────────────────────────────────────────────────────────────────

def apply_threshold(gray: np.ndarray, method: str) -> np.ndarray:
    """Return a binary mask using the chosen thresholding strategy."""
    if method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "otsu_inv":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
    elif method == "adaptive_inv":
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=2,
        )
    else:
        raise ValueError(f"Unknown thresh_method: {method!r}")
    return thresh


# ──────────────────────────────────────────────────────────────────────────────
# Contour detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_contours(gray: np.ndarray, p: SimpleNamespace) -> list:
    """
    Full contour detection pipeline targeting barnacle rims (ring shapes):

    1. Threshold → morphological OPEN (removes noise without filling rings)
    2. findContours with RETR_CCOMP — returns a two-level hierarchy where
       outer contours are at hierarchy level 0 and their interior holes are
       at level 1 (children). A barnacle rim appears as an outer contour
       with exactly one child hole.
    3. Keep only contours that have a child (ring topology).
    4. Filter by rim area and annularity (rim area / bounding circle area).
       Thin rings score low on annularity; filled blobs score close to 1.0.
    """
    thresh = apply_threshold(gray, p.thresh_method)

    # OPEN = erode then dilate: kills speckle without bridging the ring gap
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_kernel, p.morph_kernel))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=p.morph_iters)

    # RETR_CCOMP: two-level hierarchy — level-0 outer contours, level-1 holes
    # hierarchy shape: (1, N, 4) where each entry is [next, prev, child, parent]
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]  # unwrap the outer dimension → shape (N, 4)
    accepted  = []

    for i, cnt in enumerate(contours):
        # Only consider top-level contours (no parent)
        if hierarchy[i][3] != -1:
            continue

        # Ring topology check: must have at least one child hole
        if hierarchy[i][2] == -1:
            continue

        area = cv2.contourArea(cnt)
        if not (p.min_area <= area <= p.max_area):
            continue

        # Annularity: rim area relative to the area of the bounding circle
        # Thin rings → low value; filled discs → close to 1.0
        _, radius = cv2.minEnclosingCircle(cnt)
        bounding_circle_area = np.pi * radius ** 2
        if bounding_circle_area == 0:
            continue
        annularity = area / bounding_circle_area
        if not (p.min_annularity <= annularity <= p.max_annularity):
            continue

        accepted.append(cnt)

    return accepted


# ──────────────────────────────────────────────────────────────────────────────
# Annotation
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(image: np.ndarray, contours: list,
                    raw_count: int, adjusted: float,
                    params_label: str) -> np.ndarray:
    out = image.copy()

    # Draw each contour outline and mark its centroid
    for cnt in contours:
        cv2.drawContours(out, [cnt], -1, (0, 210, 255), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(out, (cx, cy), 2, (0, 210, 255), -1)

    h, w = out.shape[:2]
    bar_h = 52
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.60, out, 0.40, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, params_label,                           (8, 16), font, 0.40, (180, 180, 180), 1)
    cv2.putText(out, f"raw={raw_count}  adj={adjusted:.1f}", (8, 40), font, 0.60, (0, 210, 255),   2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Grid sweep
# ──────────────────────────────────────────────────────────────────────────────

def build_param_combos() -> list:
    """Expand PARAM_GRID into a flat list of SimpleNamespace objects."""
    # Pull out the paired params so they expand as units, not crossed
    area_pairs        = PARAM_GRID.get("area_pair",        [(100, 2000)])
    annularity_pairs  = PARAM_GRID.get("annularity_pair",  [(0.1, 0.6)])
    scalar_grid       = {k: v for k, v in PARAM_GRID.items()
                         if k not in ("area_pair", "annularity_pair")}

    keys   = list(scalar_grid.keys())
    values = list(scalar_grid.values())

    combos = []
    for ap, annp in itertools.product(area_pairs, annularity_pairs):
        for combo in itertools.product(*values):
            p = SimpleNamespace(**FIXED)
            for k, v in zip(keys, combo):
                setattr(p, k, v)
            p.min_area,        p.max_area        = ap
            p.min_annularity,  p.max_annularity  = annp
            combos.append(p)
    return combos


def short_tag(p: SimpleNamespace) -> str:
    """Build a compact filename tag from the swept parameters."""
    return (
        f"th={p.thresh_method}"
        f"_k={p.morph_kernel}"
        f"_i={p.morph_iters}"
        f"_a={p.min_area}-{p.max_area}"
        f"_ann={p.min_annularity}-{p.max_annularity}"
    )


def run_sweep(image_path: str, out_dir: Path,
              adjustment: float, max_count: int) -> list:
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path}")

    gray   = preprocess(image)
    combos = build_param_combos()
    total  = len(combos)
    print(f"[→] {total} parameter combinations to test on '{image_path}'\n")

    results = []
    for i, p in enumerate(combos, 1):
        p.adjustment = adjustment
        t0 = time.perf_counter()

        contours = detect_contours(gray, p)

        raw      = len(contours)
        adjusted = raw * p.adjustment
        elapsed  = (time.perf_counter() - t0) * 1000

        tag  = short_tag(p)
        bar  = "█" * int(30 * i / total) + "░" * (30 - int(30 * i / total))

        if raw >= max_count:
            print(f"  [{bar}] {i:>4}/{total}  raw={raw:>4}  SKIPPED (>={max_count})  {tag}", end="\r")
            continue

        stem      = Path(image_path).stem
        filename  = f"{stem}__{tag}.jpg"
        out_path  = out_dir / filename

        label = (f"th={p.thresh_method} k={p.morph_kernel} i={p.morph_iters} "
                 f"area={p.min_area}-{p.max_area} "
                 f"ann={p.min_annularity}-{p.max_annularity}")
        annotated = draw_detections(image, contours, raw, adjusted, label)
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

        r = RunResult(
            thresh_method  = p.thresh_method,
            morph_kernel   = p.morph_kernel,
            morph_iters    = p.morph_iters,
            min_area       = p.min_area,
            max_area       = p.max_area,
            min_annularity = p.min_annularity,
            max_annularity = p.max_annularity,
            contours       = raw,
            adjusted       = adjusted,
            out_file       = filename,
            elapsed_ms     = round(elapsed, 1),
        )
        results.append(r)

        print(f"  [{bar}] {i:>4}/{total}  raw={raw:>4}  SAVED  {tag}", end="\r")

    print()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CSV + contact-sheet summary
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(results: list, out_dir: Path) -> Path:
    path = out_dir / "contour_tuning_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    print(f"[✓] CSV saved → {path}")
    return path


def save_contact_sheet(results: list, out_dir: Path, cols: int = 6):
    """
    Stitch every annotated thumbnail into a single contact-sheet image,
    sorted by contour count ascending so you can scan from sparse→dense.
    """
    results_sorted = sorted(results, key=lambda r: r.contours)
    thumb_w, thumb_h = 320, 240
    thumbs = []

    for r in results_sorted:
        img = cv2.imread(str(out_dir / r.out_file))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        cv2.putText(thumb, f"raw={r.contours}", (6, thumb_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1)
        thumbs.append(thumb)

    if not thumbs:
        print("[!] No thumbnails to stitch — contact sheet skipped.")
        return

    rows_needed = (len(thumbs) + cols - 1) // cols
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    while len(thumbs) % cols != 0:
        thumbs.append(blank)

    rows_imgs = [np.hstack(thumbs[i * cols: (i + 1) * cols]) for i in range(rows_needed)]
    sheet = np.vstack(rows_imgs)
    path  = out_dir / "contour_contact_sheet.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"[✓] Contact sheet saved → {path}  ({rows_needed} rows × {cols} cols)")


def print_summary(results: list, total_ran: int, max_count: int):
    skipped = total_ran - len(results)
    print(f"\n── Run summary  ({len(results)} saved / {skipped} skipped ≥{max_count}) ────")
    counts = sorted(set(r.contours for r in results))
    print("── Count distribution ─────────────────────────")
    for c in counts:
        n   = sum(1 for r in results if r.contours == c)
        bar = "▮" * min(n, 40)
        print(f"  raw={c:>4}  ({n:>3} combos)  {bar}")

    if results:
        best_mid = sorted(results, key=lambda r: r.contours)[len(results) // 2]
        print(f"\n── Median-count combo ──────────────────────────")
        print(f"  raw={best_mid.contours}  →  {best_mid.out_file}")
    print("────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Grid-search contour detector parameters and save tagged outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",      required=True,
                   help="Path to cropped barnacle image (Stage 1 output)")
    p.add_argument("--out_dir",    default="media/contour_runs",
                   help="Directory to write annotated images, CSV, and contact sheet")
    p.add_argument("--adjustment", type=float, default=1.0,
                   help="Adjustment constant applied to every run's raw count")
    p.add_argument("--max_count",  type=int,   default=200,
                   help="Skip and don't save runs where raw detections >= this value")
    p.add_argument("--cols",       type=int,   default=6,
                   help="Columns in the contact sheet")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_sweep(args.image, out_dir, args.adjustment, args.max_count)

    if not results:
        print("[!] No runs were saved — try raising --max_count or adjusting PARAM_GRID.")
        return

    save_csv(results, out_dir)
    save_contact_sheet(results, out_dir, cols=args.cols)
    print_summary(results, total_ran=len(build_param_combos()), max_count=args.max_count)


if __name__ == "__main__":
    main()
