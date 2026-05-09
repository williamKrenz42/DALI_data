import cv2
import numpy as np
import os

# ── Input ─────────────────────────────────────────────────────────────────────
INPUT_PATH   = "media/img1.png"
OUTPUT_DIR   = "output_tuning"

# ── HSV colour range (keep fixed while tuning lines) ─────────────────────────
LOWER_GREEN = np.array([60,  30,  10])
UPPER_GREEN = np.array([105, 255, 120])

# ── Parameter grid to sweep ───────────────────────────────────────────────────
HOUGH_THRESHOLDS = [50, 80, 120]        # votes needed to accept a line
MIN_LINE_LENGTHS = [200, 300, 400]       # shortest valid segment (px)
MAX_LINE_GAPS    = [110, 120, 130, 200]          # max gap within a segment (px)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_green_mask(img):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask   = cv2.dilate(mask, kernel, iterations=2)
    mask   = cv2.erode (mask, kernel, iterations=2)
    return mask


def detect_and_draw(img, mask, threshold, min_length, max_gap):
    """Run HoughLinesP and draw results onto a copy of the image."""
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho          = 1,
        theta        = np.pi / 180,
        threshold    = threshold,
        minLineLength= min_length,
        maxLineGap   = max_gap,
    )

    out = img.copy()
    n_lines = 0

    if lines is not None:
        n_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red lines

    # Burn parameter info into the image so it's readable without the filename
    label = f"thresh={threshold}  minLen={min_length}  maxGap={max_gap}  lines={n_lines}"
    cv2.putText(out, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

    return out


def make_filename(threshold, min_length, max_gap):
    """e.g. thresh080_minLen150_maxGap015.png"""
    return f"thresh{threshold:03d}_minLen{min_length:03d}_maxGap{max_gap:03d}.png"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {INPUT_PATH}")

    mask = build_green_mask(img)

    total = len(HOUGH_THRESHOLDS) * len(MIN_LINE_LENGTHS) * len(MAX_LINE_GAPS)
    print(f"Sweeping {total} parameter combinations → {OUTPUT_DIR}/")

    count = 0
    for threshold in HOUGH_THRESHOLDS:
        for min_length in MIN_LINE_LENGTHS:
            for max_gap in MAX_LINE_GAPS:
                result   = detect_and_draw(img, mask, threshold, min_length, max_gap)
                filename = make_filename(threshold, min_length, max_gap)
                out_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(out_path, result)
                count += 1
                print(f"  [{count}/{total}] {filename}")

    print("Done.")


if __name__ == "__main__":
    main()