"""
barnacle_counter.py
--------------------
Counts barnacles in a pre-cropped image using OpenCV circle / ellipse detection,
then applies an adjustment constant to correct for systematic over- or under-counting.

Usage:
    python barnacle_counter.py --image path/to/cropped.jpg [options]

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# DEFAULT TUNING PARAMETERS  (edit freely)
# ──────────────────────────────────────────────
DEFAULTS = dict(
    # --- Hough circle params ---
    dp=1,           # Inverse ratio of accumulator resolution to image resolution
    min_dist=25,      # Minimum distance between detected circle centres (px)
    param1=50,        # Upper Canny threshold (lower = param1/2)
    param2=35,        # Accumulator threshold – lower → more (false) circles
    min_radius=6,     # Smallest barnacle radius to look for (px)
    max_radius=40,    # Largest barnacle radius to look for (px)

    # --- Ellipse fallback ---
    use_ellipse=True,        # Also run contour-based ellipse detection?
    ellipse_min_area=200,     # Min contour area to consider (px²)
    ellipse_max_area=15000,   # Max contour area to consider
    ellipse_aspect_ratio=1.1, # Max width/height ratio to still call it an ellipse

    # --- Adjustment constant ---
    adjustment=1.0,   # Multiply raw count by this value (e.g. 0.9 for 10% over-count)

    # --- Output ---
    output=None,      # Save annotated image to this path (None = display only)
    debug=False,      # Show intermediate processing steps
)


def preprocess(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """Convert to greyscale, denoise, and enhance edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter keeps edges sharp while smoothing texture noise
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # CLAHE boosts local contrast – helps separate barnacles from substrate
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    if debug:
        cv2.imshow("Preprocessed", enhanced)
        cv2.waitKey(0)
    return enhanced


def detect_circles(gray: np.ndarray, args) -> list[tuple[int, int, int]]:
    """Run Hough Circle Transform and return list of (x, y, r) tuples."""
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=args.dp,
        minDist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        minRadius=args.min_radius,
        maxRadius=args.max_radius,
    )
    if circles is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in circles[0]]


def detect_ellipses(gray: np.ndarray, args) -> list[tuple]:
    """
    Fallback: find ellipse-shaped contours for barnacles that are
    partially occluded or non-circular.
    Returns a list of cv2.RotatedRect tuples (centre, axes, angle).
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological closing fills small gaps inside barnacle rims
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (args.ellipse_min_area <= area <= args.ellipse_max_area):
            continue
        if len(cnt) < 5:           # fitEllipse needs ≥ 5 points
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, mi), angle = ellipse
        if mi == 0:
            continue
        aspect = ma / mi
        if aspect <= args.ellipse_aspect_ratio:
            ellipses.append(ellipse)
    return ellipses


def draw_detections(
    image: np.ndarray,
    circles: list,
    ellipses: list,
    raw_count: int,
    adjusted_count: float,
    adjustment: float,
) -> np.ndarray:
    """Annotate the image with circles, ellipses, and count overlay."""
    out = image.copy()

    # Draw circles
    for (x, y, r) in circles:
        cv2.circle(out, (x, y), r, (0, 230, 120), 2)
        cv2.circle(out, (x, y), 2, (0, 230, 120), -1)

    # Draw ellipses (different colour so they're distinguishable)
    for ellipse in ellipses:
        cv2.ellipse(out, ellipse, (60, 180, 255), 2)

    # Overlay count info
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, f"Raw detections: {raw_count}", (12, h - 50), font, 0.65, (200, 200, 200), 1)
    cv2.putText(
        out,
        f"Adjusted count: {adjusted_count:.1f}  (x{adjustment})",
        (12, h - 18),
        font, 0.75, (0, 230, 120), 2,
    )
    return out


def count_barnacles(image_path: str, args) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"[ERROR] Could not read image: {image_path}")

    gray = preprocess(image, debug=args.debug)

    circles  = detect_circles(gray, args)
    ellipses = detect_ellipses(gray, args) if args.use_ellipse else []

    # Deduplicate: if a circle centre falls inside an already-found ellipse, skip it
    # (simple centroid-distance guard to avoid double-counting)
    if ellipses:
        ellipse_centres = np.array([(int(e[0][0]), int(e[0][1])) for e in ellipses])
        filtered_circles = []
        for (cx, cy, r) in circles:
            dists = np.linalg.norm(ellipse_centres - np.array([cx, cy]), axis=1)
            if dists.min() > r:          # not already covered by an ellipse
                filtered_circles.append((cx, cy, r))
        circles = filtered_circles

    raw_count     = len(circles) + len(ellipses)
    adjusted_count = raw_count * args.adjustment

    annotated = draw_detections(image, circles, ellipses, raw_count, adjusted_count, args.adjustment)

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"[✓] Annotated image saved → {args.output}")
    else:
        cv2.imshow("Barnacle Counter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    result = {
        "image":          image_path,
        "circles_found":  len(circles),
        "ellipses_found": len(ellipses),
        "raw_count":      raw_count,
        "adjustment":     args.adjustment,
        "adjusted_count": adjusted_count,
    }
    return result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Count barnacles in a cropped image using circle / ellipse detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",       required=True,  help="Path to the cropped barnacle image")

    # Hough params
    p.add_argument("--dp",          type=float, default=DEFAULTS["dp"])
    p.add_argument("--min_dist",    type=int,   default=DEFAULTS["min_dist"],
                   help="Min distance between circle centres (px)")
    p.add_argument("--param1",      type=int,   default=DEFAULTS["param1"],
                   help="Canny upper threshold")
    p.add_argument("--param2",      type=int,   default=DEFAULTS["param2"],
                   help="Accumulator threshold – lower = more detections")
    p.add_argument("--min_radius",  type=int,   default=DEFAULTS["min_radius"])
    p.add_argument("--max_radius",  type=int,   default=DEFAULTS["max_radius"])

    # Ellipse fallback
    p.add_argument("--use_ellipse", action="store_true", default=DEFAULTS["use_ellipse"],
                   help="Also detect ellipse-shaped barnacles via contours")
    p.add_argument("--ellipse_min_area",    type=int,   default=DEFAULTS["ellipse_min_area"])
    p.add_argument("--ellipse_max_area",    type=int,   default=DEFAULTS["ellipse_max_area"])
    p.add_argument("--ellipse_aspect_ratio",type=float, default=DEFAULTS["ellipse_aspect_ratio"])

    # Adjustment
    p.add_argument("--adjustment",  type=float, default=DEFAULTS["adjustment"],
                   help="Multiply raw count by this constant (e.g. 0.85)")

    # Output
    p.add_argument("--output",      default=DEFAULTS["output"],
                   help="Save annotated image here instead of displaying it")
    p.add_argument("--debug",       action="store_true", default=DEFAULTS["debug"],
                   help="Show intermediate processing windows")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = count_barnacles(args.image, args)

    print("\n── Barnacle Count Report ──────────────────")
    print(f"  Image            : {result['image']}")
    print(f"  Circles detected : {result['circles_found']}")
    print(f"  Ellipses detected: {result['ellipses_found']}")
    print(f"  Raw total        : {result['raw_count']}")
    print(f"  Adjustment (×)   : {result['adjustment']}")
    print(f"  Final count      : {result['adjusted_count']:.1f}")
    print("───────────────────────────────────────────\n")