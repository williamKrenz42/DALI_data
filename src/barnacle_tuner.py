"""
barnacle_tuner.py
------------------
Grid-searches Hough / ellipse parameters, saves one annotated image per
combination with a tagged filename, and writes a CSV + summary image so
you can compare results at a glance.

Usage:
    python barnacle_tuner.py --image cropped.jpg
    python barnacle_tuner.py --image cropped.jpg --out_dir tuning_runs/

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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace


# ══════════════════════════════════════════════════════════════════════════════
#  PARAM GRID  –  edit this to control what gets swept
#  Each key maps to a list of values to try.
#  Total runs = product of all list lengths.
# ══════════════════════════════════════════════════════════════════════════════
PARAM_GRID = {
    # Hough accumulator threshold  (lower → more / noisier detections)
    "param2":      [15, 25, 35, 50],
    # Canny upper threshold
    "param1":      [50, 80, 120],
    # Min distance between circle centres (px)
    "min_dist":    [15, 25, 40],
    # Radius bounds  (px)  — kept as paired tuples so they always make sense
    "radius_pair": [(8, 50), (10, 70), (6, 40)],
    # Hough accumulator resolution  (1.0 = full res, 1.5 = downscaled)
    "dp":          [1.0, 1.2, 1.5],
}

# Fixed params not being swept
FIXED = dict(
    adjustment      = 1.0,
    use_ellipse     = False,
    ellipse_min_area    = 200,
    ellipse_max_area    = 15000,
    ellipse_aspect_ratio= 2.5,
    debug           = False,
)

# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    param2:     int
    param1:     int
    min_dist:   int
    min_radius: int
    max_radius: int
    dp:         float
    circles:    int
    ellipses:   int
    raw_count:  int
    adjusted:   float
    out_file:   str
    elapsed_ms: float


# ──────────────────────────────────────────────
# Reuse core functions from barnacle_counter.py
# (copied inline so the tuner is self-contained)
# ──────────────────────────────────────────────

def preprocess(image: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(blurred)


def detect_circles(gray: np.ndarray, p) -> list:
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=p.dp, minDist=p.min_dist,
        param1=p.param1, param2=p.param2,
        minRadius=p.min_radius, maxRadius=p.max_radius,
    )
    if circles is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in circles[0]]


def detect_ellipses(gray: np.ndarray, p) -> list:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (p.ellipse_min_area <= area <= p.ellipse_max_area):
            continue
        if len(cnt) < 5:
            continue
        e = cv2.fitEllipse(cnt)
        (_, _), (ma, mi), _ = e
        if mi == 0 or (ma / mi) > p.ellipse_aspect_ratio:
            continue
        ellipses.append(e)
    return ellipses


def draw_detections(image, circles, ellipses, raw_count, adjusted, params_label):
    out = image.copy()
    for (x, y, r) in circles:
        cv2.circle(out, (x, y), r, (0, 230, 120), 2)
        cv2.circle(out, (x, y), 2,  (0, 230, 120), -1)
    for e in ellipses:
        cv2.ellipse(out, e, (60, 180, 255), 2)

    h, w = out.shape[:2]
    # Semi-transparent header bar for params
    bar_h = 52
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.60, out, 0.40, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, params_label,          (8, 16),  font, 0.42, (180, 180, 180), 1)
    cv2.putText(out, f"raw={raw_count}  adj={adjusted:.1f}",
                (8, 40), font, 0.60, (0, 230, 120), 2)
    return out


# ──────────────────────────────────────────────
# Grid sweep
# ──────────────────────────────────────────────

def build_param_combos() -> list[SimpleNamespace]:
    """Expand PARAM_GRID into a flat list of SimpleNamespace objects."""
    # Separate radius_pair from the rest
    grid_without_radius = {k: v for k, v in PARAM_GRID.items() if k != "radius_pair"}
    radius_pairs = PARAM_GRID.get("radius_pair", [(8, 60)])

    keys   = list(grid_without_radius.keys())
    values = list(grid_without_radius.values())

    combos = []
    for rp in radius_pairs:
        for combo in itertools.product(*values):
            p = SimpleNamespace(**FIXED)
            for k, v in zip(keys, combo):
                setattr(p, k, v)
            p.min_radius, p.max_radius = rp
            combos.append(p)
    return combos


def short_tag(p: SimpleNamespace) -> str:
    """Build a compact filename tag from the swept parameters."""
    return (
        f"p1={p.param1}"
        f"_p2={p.param2}"
        f"_md={p.min_dist}"
        f"_r={p.min_radius}-{p.max_radius}"
        f"_dp={p.dp}"
    )


def run_sweep(image_path: str, out_dir: Path, adjustment: float) -> list[RunResult]:
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

        circles  = detect_circles(gray, p)
        ellipses = detect_ellipses(gray, p) if p.use_ellipse else []

        # Dedup circles that overlap with ellipse centres
        if ellipses:
            ec = np.array([(int(e[0][0]), int(e[0][1])) for e in ellipses])
            circles = [
                c for c in circles
                if np.linalg.norm(ec - np.array([c[0], c[1]]), axis=1).min() > c[2]
            ]

        raw       = len(circles) + len(ellipses)
        adjusted  = raw * p.adjustment
        elapsed   = (time.perf_counter() - t0) * 1000

        tag      = short_tag(p)
        stem     = Path(image_path).stem
        filename = f"{stem}__{tag}.jpg"
        out_path = out_dir / filename

        label = (f"p1={p.param1} p2={p.param2} md={p.min_dist} "
                 f"r={p.min_radius}-{p.max_radius} dp={p.dp}")
        annotated = draw_detections(image, circles, ellipses, raw, adjusted, label)
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

        r = RunResult(
            param2=p.param2, param1=p.param1, min_dist=p.min_dist,
            min_radius=p.min_radius, max_radius=p.max_radius, dp=p.dp,
            circles=len(circles), ellipses=len(ellipses),
            raw_count=raw, adjusted=adjusted,
            out_file=filename, elapsed_ms=round(elapsed, 1),
        )
        results.append(r)

        bar = "█" * int(30 * i / total) + "░" * (30 - int(30 * i / total))
        print(f"  [{bar}] {i:>4}/{total}  raw={raw:>4}  {tag}", end="\r")

    print()  # newline after progress bar
    return results


# ──────────────────────────────────────────────
# CSV + contact-sheet summary
# ──────────────────────────────────────────────

def save_csv(results: list[RunResult], out_dir: Path):
    path = out_dir / "tuning_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    print(f"[✓] CSV saved → {path}")
    return path


def save_contact_sheet(results: list[RunResult], out_dir: Path, cols: int = 6):
    """
    Stitch every annotated thumbnail into a single contact-sheet image
    sorted by raw_count ascending so you can scan from sparse→dense.
    """
    results_sorted = sorted(results, key=lambda r: r.raw_count)
    thumbs = []
    thumb_w, thumb_h = 320, 240

    for r in results_sorted:
        img_path = out_dir / r.out_file
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        # Label with count
        cv2.putText(thumb, f"raw={r.raw_count}", (6, thumb_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 120), 1)
        thumbs.append(thumb)

    if not thumbs:
        return

    rows_needed = (len(thumbs) + cols - 1) // cols
    # Pad to full grid
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    while len(thumbs) % cols != 0:
        thumbs.append(blank)

    rows_imgs = [
        np.hstack(thumbs[i * cols: (i + 1) * cols])
        for i in range(rows_needed)
    ]
    sheet = np.vstack(rows_imgs)
    path  = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"[✓] Contact sheet saved → {path}  ({rows_needed} rows × {cols} cols)")


def print_summary(results: list[RunResult]):
    counts = sorted(set(r.raw_count for r in results))
    print("\n── Count distribution ─────────────────────────")
    for c in counts:
        n   = sum(1 for r in results if r.raw_count == c)
        bar = "▮" * n
        print(f"  raw={c:>4}  ({n:>3} combos)  {bar}")

    best_mid = sorted(results, key=lambda r: r.raw_count)[len(results) // 2]
    print(f"\n── Median-count combo ──────────────────────────")
    print(f"  raw={best_mid.raw_count}  →  {best_mid.out_file}")
    print("────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Grid-search barnacle detector parameters and save tagged outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",      required=True, help="Path to cropped barnacle image")
    p.add_argument("--out_dir",    default="tuning_output",
                   help="Directory to write annotated images, CSV, and contact sheet")
    p.add_argument("--adjustment", type=float, default=1.0,
                   help="Adjustment constant applied to every run")
    p.add_argument("--cols",       type=int,   default=6,
                   help="Columns in the contact sheet")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_sweep(args.image, out_dir, args.adjustment)
    save_csv(results, out_dir)
    save_contact_sheet(results, out_dir, cols=args.cols)
    print_summary(results)


if __name__ == "__main__":
    main()