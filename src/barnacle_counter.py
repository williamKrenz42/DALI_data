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
import base64
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from types import SimpleNamespace

try:
    from crop import crop_image
except ModuleNotFoundError:
    from src.crop import crop_image


# ──────────────────────────────────────────────
# DEFAULT TUNING PARAMETERS  (edit freely)
# ──────────────────────────────────────────────
DEFAULTS = dict(
    # --- Hough circle params ---
    dp=1,           # Inverse ratio of accumulator resolution to image resolution
    min_dist=20,      # Minimum distance between detected circle centres (px)
    param1=40,        # Upper Canny threshold (lower = param1/2)
    param2=20,        # Accumulator threshold – lower → more (false) circles
    min_radius=10,     # Smallest barnacle radius to look for (px)
    max_radius=25,    # Largest barnacle radius to look for (px)

    # --- Cluster filter ---
    use_cluster_filter=False,    # Drop circles isolated from other circle detections?
    cluster_radius=50,           # Neighbour search radius around each circle centre (px)
    min_cluster_neighbors=1,     # Required neighbouring circle centres inside cluster_radius

    # --- Ellipse fallback ---
    use_ellipse=True,        # Also run contour-based ellipse detection?
    ellipse_min_area=50,     # Min contour area to consider (px²)
    ellipse_max_area=1000,   # Max contour area to consider
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


def filter_isolated_circles(
    circles: list[tuple[int, int, int]],
    radius: int,
    min_neighbors: int,
) -> list[tuple[int, int, int]]:
    """
    Keep only circles with enough nearby circle centres.

    This uses a small spatial hash instead of comparing every circle to every
    other circle, so it stays cheap even when Hough produces many candidates.
    """
    if len(circles) <= min_neighbors or radius <= 0 or min_neighbors <= 0:
        return circles

    cell_size = radius
    radius_sq = radius * radius
    grid: dict[tuple[int, int], list[int]] = {}

    for i, (x, y, _) in enumerate(circles):
        cell = (x // cell_size, y // cell_size)
        grid.setdefault(cell, []).append(i)

    kept = []
    for i, (x, y, r) in enumerate(circles):
        cell_x = x // cell_size
        cell_y = y // cell_size
        neighbors = 0

        for gx in range(cell_x - 1, cell_x + 2):
            for gy in range(cell_y - 1, cell_y + 2):
                for j in grid.get((gx, gy), []):
                    if i == j:
                        continue
                    nx, ny, _ = circles[j]
                    dx = x - nx
                    dy = y - ny
                    if dx * dx + dy * dy <= radius_sq:
                        neighbors += 1
                        if neighbors >= min_neighbors:
                            kept.append((x, y, r))
                            break
                if neighbors >= min_neighbors:
                    break
            if neighbors >= min_neighbors:
                break

    return kept


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


def count_barnacles_in_image(image: np.ndarray, args, image_path: str = "") -> dict:
    """Run detection on an already-loaded/cropped image and return counts + overlay."""
    gray = preprocess(image, debug=args.debug)

    circles  = detect_circles(gray, args)
    ellipses = detect_ellipses(gray, args) if args.use_ellipse else []

    if args.use_cluster_filter:
        circles = filter_isolated_circles(
            circles,
            radius=args.cluster_radius,
            min_neighbors=args.min_cluster_neighbors,
        )

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

    result = {
        "image":          image_path,
        "circles_found":  len(circles),
        "ellipses_found": len(ellipses),
        "raw_count":      raw_count,
        "adjustment":     args.adjustment,
        "adjusted_count": adjusted_count,
        "annotated":      annotated,
    }
    return result


def count_barnacles(image_path: str, args) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"[ERROR] Could not read image: {image_path}")

    result = count_barnacles_in_image(image, args, image_path=image_path)
    annotated = result["annotated"]

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"[✓] Annotated image saved → {args.output}")
    else:
        cv2.imshow("Barnacle Counter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def default_args() -> SimpleNamespace:
    """Build an argparse-like namespace from DEFAULTS for GUI runs."""
    return SimpleNamespace(**DEFAULTS)


def cv_to_photo_image(image: np.ndarray, max_width: int = 1000, max_height: int = 700) -> tk.PhotoImage:
    """Convert an OpenCV BGR image into a Tk PhotoImage, resized to fit."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".png", rgb)
    if not ok:
        raise ValueError("Could not encode annotated image for display.")
    png_data = base64.b64encode(encoded.tobytes())
    return tk.PhotoImage(data=png_data)


class BarnacleCounterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Barnacle Counter")
        self.root.geometry("1100x820")
        self.root.minsize(760, 560)
        self.photo = None

        self.status_var = tk.StringVar(value="Choose an image to crop and count.")
        self.count_var = tk.StringVar(value="Estimated count: --")

        top = tk.Frame(root, padx=14, pady=12)
        top.pack(fill=tk.X)

        self.upload_button = tk.Button(top, text="Upload Image", command=self.choose_image)
        self.upload_button.pack(side=tk.LEFT)

        tk.Label(top, textvariable=self.count_var, font=("Helvetica", 16, "bold")).pack(side=tk.LEFT, padx=18)

        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w", padx=14)
        self.status_label.pack(fill=tk.X)

        self.image_label = tk.Label(root, bg="#202020")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose barnacle image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.upload_button.config(state=tk.DISABLED)
        self.count_var.set("Estimated count: --")
        self.status_var.set("Cropping grid and counting barnacles...")
        threading.Thread(target=self.process_image, args=(path,), daemon=True).start()

    def process_image(self, path: str):
        try:
            roi = crop_image(path, "media/output.jpg")
            args = default_args()
            result = count_barnacles_in_image(roi, args, image_path=path)
            cv2.imwrite("media/final.jpg", result["annotated"])
        except Exception as exc:
            self.root.after(0, self.show_error, exc)
            return

        self.root.after(0, self.show_result, result)

    def show_result(self, result: dict):
        self.upload_button.config(state=tk.NORMAL)
        self.count_var.set(f"Estimated count: {result['adjusted_count']:.1f}")
        self.status_var.set(
            f"Circles: {result['circles_found']}   "
            f"Ellipses: {result['ellipses_found']}   "
            "Saved crop to media/output.jpg and overlay to media/final.jpg"
        )

        self.photo = cv_to_photo_image(result["annotated"])
        self.image_label.config(image=self.photo)

    def show_error(self, exc: Exception):
        self.upload_button.config(state=tk.NORMAL)
        self.status_var.set("Processing failed.")
        messagebox.showerror("Barnacle Counter", str(exc))


def launch_gui():
    root = tk.Tk()
    BarnacleCounterApp(root)
    root.mainloop()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Count barnacles in a cropped image using circle / ellipse detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",       help="Path to a pre-cropped barnacle image. Omit to launch the GUI.")

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

    # Cluster filter
    p.add_argument("--use_cluster_filter", action="store_true", default=DEFAULTS["use_cluster_filter"],
                   help="Drop circles that do not have enough neighbouring circle detections")
    p.add_argument("--cluster_radius", type=int, default=DEFAULTS["cluster_radius"],
                   help="Neighbour search radius for --use_cluster_filter (px)")
    p.add_argument("--min_cluster_neighbors", type=int, default=DEFAULTS["min_cluster_neighbors"],
                   help="Required neighbours inside --cluster_radius")

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

    if not args.image:
        launch_gui()
        sys.exit(0)

    result = count_barnacles(args.image, args)

    print("\n── Barnacle Count Report ──────────────────")
    print(f"  Image            : {result['image']}")
    print(f"  Circles detected : {result['circles_found']}")
    print(f"  Ellipses detected: {result['ellipses_found']}")
    print(f"  Raw total        : {result['raw_count']}")
    print(f"  Adjustment (×)   : {result['adjustment']}")
    print(f"  Final count      : {result['adjusted_count']:.1f}")
    print("───────────────────────────────────────────\n")
