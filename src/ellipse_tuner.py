"""
ellipse_tuner.py
-----------------
Grid-searches contour-based ellipse detection parameters, saves one annotated
image per combination with a tagged filename, and writes a CSV + contact sheet
so you can compare results at a glance.

Usage:
    python src/ellipse_tuner.py --image media/output.jpg
    python src/ellipse_tuner.py --image media/output.jpg --out_dir media/ellipse_runs/

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
    "thresh_method": ["otsu", "adaptive", "otsu_inv"],
    "morph_kernel":  [3, 5],
    "morph_iters":   [1, 2],
    "area_pair":     [(50, 1000), (100, 2000), (200, 3000)],
    # Max major/minor axis ratio — raised to accept barnacles that are
    # noticeably non-circular or partially occluded
    "max_aspect":    [3.0, 4.0, 6.0],
    # Solidity = contour_area / convex_hull_area.
    # Values close to 1.0 = compact/convex shapes (good barnacle proxy).
    # A low minimum allows rougher, partially-occluded outlines through;
    # a high minimum keeps only clean, well-defined rims.
    "min_solidity":  [0.5, 0.65, 0.8],
}


# Fixed params not being swept
FIXED = dict(
    adjustment = 1.0,
    debug      = False,
)

# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    thresh_method: str
    morph_kernel:  int
    morph_iters:   int
    min_area:      int
    max_area:      int
    max_aspect:    float
    min_solidity:  float
    ellipses:      int
    adjusted:      float
    out_file:      str
    elapsed_ms:    float


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing (identical to barnacle_counter.py)
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
    else:
        raise ValueError(f"Unknown thresh_method: {method!r}")
    return thresh


# ──────────────────────────────────────────────────────────────────────────────
# Ellipse detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_ellipses(gray: np.ndarray, p: SimpleNamespace) -> list:
    """
    Full ellipse detection pipeline:
        threshold → morphological close → find contours → fitEllipse → filter
    Returns a list of cv2.RotatedRect tuples (centre, axes, angle).
    """
    thresh = apply_threshold(gray, p.thresh_method)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_kernel, p.morph_kernel))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=p.morph_iters)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (p.min_area <= area <= p.max_area):
            continue
        if len(cnt) < 5:            # fitEllipse requires ≥ 5 points
            continue

        # Solidity: how much of the convex hull is filled by the contour.
        # Rejects jagged merged blobs while accepting rough individual rims.
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        if (area / hull_area) < p.min_solidity:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (_, _), (ma, mi), _ = ellipse
        if mi == 0:
            continue
        if (ma / mi) > p.max_aspect:
            continue
        ellipses.append(ellipse)

    return ellipses


# ──────────────────────────────────────────────────────────────────────────────
# Annotation
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(image: np.ndarray, ellipses: list,
                    raw_count: int, adjusted: float,
                    params_label: str) -> np.ndarray:
    out = image.copy()

    for e in ellipses:
        cv2.ellipse(out, e, (60, 180, 255), 2)
        cx, cy = int(e[0][0]), int(e[0][1])
        cv2.circle(out, (cx, cy), 2, (60, 180, 255), -1)

    h, w = out.shape[:2]
    bar_h = 52
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.60, out, 0.40, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, params_label,                       (8, 16), font, 0.40, (180, 180, 180), 1)
    cv2.putText(out, f"raw={raw_count}  adj={adjusted:.1f}", (8, 40), font, 0.60, (60, 180, 255), 2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Grid sweep
# ──────────────────────────────────────────────────────────────────────────────

def build_param_combos() -> list:
    """Expand PARAM_GRID into a flat list of SimpleNamespace objects."""
    grid_without_area = {k: v for k, v in PARAM_GRID.items() if k != "area_pair"}
    area_pairs = PARAM_GRID.get("area_pair", [(200, 15000)])

    keys   = list(grid_without_area.keys())
    values = list(grid_without_area.values())

    combos = []
    for ap in area_pairs:
        for combo in itertools.product(*values):
            p = SimpleNamespace(**FIXED)
            for k, v in zip(keys, combo):
                setattr(p, k, v)
            p.min_area, p.max_area = ap
            combos.append(p)
    return combos


def short_tag(p: SimpleNamespace) -> str:
    """Build a compact filename tag from the swept parameters."""
    return (
        f"th={p.thresh_method}"
        f"_k={p.morph_kernel}"
        f"_i={p.morph_iters}"
        f"_a={p.min_area}-{p.max_area}"
        f"_ar={p.max_aspect}"
        f"_sol={p.min_solidity}"
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

        ellipses = detect_ellipses(gray, p)

        raw      = len(ellipses)
        adjusted = raw * p.adjustment
        elapsed  = (time.perf_counter() - t0) * 1000

        tag  = short_tag(p)
        stem = Path(image_path).stem
        bar  = "█" * int(30 * i / total) + "░" * (30 - int(30 * i / total))

        if raw >= max_count:
            print(f"  [{bar}] {i:>4}/{total}  raw={raw:>4}  SKIPPED (>={max_count})  {tag}", end="\r")
            continue

        filename = f"{stem}__{tag}.jpg"
        out_path = out_dir / filename

        label = (f"th={p.thresh_method} k={p.morph_kernel} i={p.morph_iters} "
                 f"area={p.min_area}-{p.max_area} ar={p.max_aspect} sol={p.min_solidity}")
        annotated = draw_detections(image, ellipses, raw, adjusted, label)
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

        r = RunResult(
            thresh_method = p.thresh_method,
            morph_kernel  = p.morph_kernel,
            morph_iters   = p.morph_iters,
            min_area      = p.min_area,
            max_area      = p.max_area,
            max_aspect    = p.max_aspect,
            min_solidity  = p.min_solidity,
            ellipses      = raw,
            adjusted      = adjusted,
            out_file      = filename,
            elapsed_ms    = round(elapsed, 1),
        )
        results.append(r)

        print(f"  [{bar}] {i:>4}/{total}  raw={raw:>4}  SAVED  {tag}", end="\r")

    print()  # newline after progress bar
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CSV + contact-sheet summary
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(results: list, out_dir: Path) -> Path:
    path = out_dir / "ellipse_tuning_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    print(f"[✓] CSV saved → {path}")
    return path


def save_contact_sheet(results: list, out_dir: Path, cols: int = 6):
    """
    Stitch every annotated thumbnail into a single contact-sheet image,
    sorted by ellipse count ascending so you can scan from sparse→dense.
    """
    results_sorted = sorted(results, key=lambda r: r.ellipses)
    thumb_w, thumb_h = 320, 240
    thumbs = []

    for r in results_sorted:
        img = cv2.imread(str(out_dir / r.out_file))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        cv2.putText(thumb, f"raw={r.ellipses}", (6, thumb_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 180, 255), 1)
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
    path  = out_dir / "ellipse_contact_sheet.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"[✓] Contact sheet saved → {path}  ({rows_needed} rows × {cols} cols)")


def print_summary(results: list, total_ran: int, max_count: int):
    skipped = total_ran - len(results)
    print(f"\n── Run summary  ({len(results)} saved / {skipped} skipped ≥{max_count}) ────")
    counts = sorted(set(r.ellipses for r in results))
    print("── Count distribution ─────────────────────────")
    for c in counts:
        n   = sum(1 for r in results if r.ellipses == c)
        bar = "▮" * min(n, 40)
        print(f"  raw={c:>4}  ({n:>3} combos)  {bar}")

    if results:
        best_mid = sorted(results, key=lambda r: r.ellipses)[len(results) // 2]
        print(f"\n── Median-count combo ──────────────────────────")
        print(f"  raw={best_mid.ellipses}  →  {best_mid.out_file}")
    print("────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Grid-search ellipse detector parameters and save tagged outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",      required=True,
                   help="Path to cropped barnacle image (Stage 1 output)")
    p.add_argument("--out_dir",    default="media/ellipse_runs",
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
