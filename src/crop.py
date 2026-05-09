import cv2
import numpy as np

# ── File paths ────────────────────────────────────────────────────────────────
INPUT_PATH  = "media/img1.png"   # change to your image path
OUTPUT_PATH = "media/output.jpg"        # change to your desired output path

# ── HSV colour range for the green grid ───────────────────────────────────────
# Tune these if your green is darker/lighter/more yellow or blue
LOWER_GREEN = np.array([75,  40,  20])
UPPER_GREEN = np.array([95, 255, 120])

# ── Hough line parameters ─────────────────────────────────────────────────────
HOUGH_THRESHOLD   = 80    # minimum votes for a line to be accepted
MIN_LINE_LENGTH   = 400   # shortest segment (px) that counts as a grid line
MAX_LINE_GAP      = 200    # max gap (px) allowed within a single line segment


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return img
 
 
def build_green_mask(img: np.ndarray) -> np.ndarray:
    """Return a binary mask isolating the green grid."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
 
    # Clean up speckle noise while keeping the thick grid lines intact
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask   = cv2.dilate(mask, kernel, iterations=2)
    mask   = cv2.erode (mask, kernel, iterations=2)
    return mask
 
 
def detect_lines(mask: np.ndarray):
    """Run Canny + HoughLinesP on the green mask."""
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho          = 1,
        theta        = np.pi / 180,
        threshold    = HOUGH_THRESHOLD,
        minLineLength= MIN_LINE_LENGTH,
        maxLineGap   = MAX_LINE_GAP,
    )
    return lines
 
 
def separate_lines(lines):
    """Split detected segments into horizontal and vertical groups.
    Any segment with absolute angle under 45 degrees is horizontal,
    45 and above is vertical."""
    horizontals, verticals = [], []
    if lines is None:
        return horizontals, verticals
 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 45:
            horizontals.append(line[0])
        else:
            verticals.append(line[0])
 
    return horizontals, verticals
 
 
def grid_bounds(horizontals, verticals, img_shape):
    """Find the outer border by:
    1. Computing the centroid of all detected line segments
    2. Filtering to only lines that span across the centroid axis
       (horizontal lines whose x range includes cx,
        vertical lines whose y range includes cy)
    3. Walking outward from the centroid to find the first line in each direction
    """
    if not horizontals or not verticals:
        raise ValueError("Not enough grid lines detected — try tuning the HSV range or Hough parameters.")
 
    # Step 1: centroid of all segments
    all_x = [(l[0] + l[2]) / 2 for l in horizontals + verticals]
    all_y = [(l[1] + l[3]) / 2 for l in horizontals + verticals]
    cx = np.mean(all_x)
    cy = np.mean(all_y)
 
    # Step 2: filter to lines that cross the centroid axis
    h_spanning = [l for l in horizontals if min(l[0], l[2]) < cx < max(l[0], l[2])]
    v_spanning = [l for l in verticals   if min(l[1], l[3]) < cy < max(l[1], l[3])]
 
    if not h_spanning or not v_spanning:
        raise ValueError("No lines spanning the centroid found — try lowering MIN_LINE_LENGTH or adjusting the HSV range.")
 
    # Step 3: walk outward from centroid to first line in each direction
    above       = [l for l in h_spanning if (l[1] + l[3]) / 2 < cy]
    below       = [l for l in h_spanning if (l[1] + l[3]) / 2 > cy]
    left_lines  = [l for l in v_spanning if (l[0] + l[2]) / 2 < cx]
    right_lines = [l for l in v_spanning if (l[0] + l[2]) / 2 > cx]
 
    if not above or not below or not left_lines or not right_lines:
        raise ValueError("Could not find bounding lines on all four sides of the centroid.")
 
    top    = int(max((l[1] + l[3]) / 2 for l in above))
    bottom = int(min((l[1] + l[3]) / 2 for l in below))
    left   = int(max((l[0] + l[2]) / 2 for l in left_lines))
    right  = int(min((l[0] + l[2]) / 2 for l in right_lines))
 
    print(f"  Centroid: ({cx:.0f}, {cy:.0f})")
    print(f"  Spanning horizontals: {len(h_spanning)}, spanning verticals: {len(v_spanning)}")
    print(f"  Bounds: top={top}, bottom={bottom}, left={left}, right={right}")
 
    src = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
    return src
 
 
def warp_roi(img: np.ndarray, src_corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the detected rectangle to a straight-on crop."""
    tl, tr, br, bl = src_corners
 
    width  = int(max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    ))
    height = int(max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    ))
 
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M   = cv2.getPerspectiveTransform(src_corners, dst)
    return cv2.warpPerspective(img, M, (width, height))
 
 
def crop_image(image_path: str, output_path: str | None = None) -> np.ndarray:
    """Run the full grid-detection crop pipeline and optionally save the ROI."""
    print(f"Loading image from: {image_path}")
    img = load_image(image_path)

    print("Building green mask...")
    mask = build_green_mask(img)

    print("Detecting grid lines...")
    lines = detect_lines(mask)
    horizontals, verticals = separate_lines(lines)
    print(f"  Found {len(horizontals)} horizontal, {len(verticals)} vertical segments")

    print("Computing bounding box...")
    src_corners = grid_bounds(horizontals, verticals, img.shape)
    print(f"  Corners: {src_corners.tolist()}")

    print("Warping ROI...")
    roi = warp_roi(img, src_corners)

    if output_path:
        print(f"Saving output to: {output_path}")
        cv2.imwrite(output_path, roi)

    return roi


def main():
    crop_image(INPUT_PATH, OUTPUT_PATH)
    print("Done.")
 
 
if __name__ == "__main__":
    main()
 
