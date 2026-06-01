"""Mask post-processing and polygon extraction for the SAM2_BigImg backend.

Everything here operates in *crop* coordinates on a single binary mask -- the
~1024px SAM2 output, before it is mapped back onto the full image. Only OpenCV
and NumPy are used (both already in the backend image), so no rebuild is needed.

Mapping polygon points from crop coordinates to full-image coordinates is left
to CropMapper.polygons_crop_to_full().
"""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Point = Tuple[float, float]
Polygon = List[Point]

# Iterations of the epsilon binary search used to honour `max_points`.
_EPSILON_SEARCH_STEPS = 12

# Defaults for polygon return formats.
# epsilon 1.0 -> pixel mode, 1px tolerance (the faithful end of the range).
POLYGON_DEFAULTS = {"epsilon": 1.0, "max_points": 100}


def filter_components(mask: np.ndarray, size_threshold: float) -> np.ndarray:
    """Keep connected components with area >= size_threshold * largest-component area.

    size_threshold == 0 keeps every blob (no filtering); == 1 keeps only the
    largest; 0.5 keeps blobs at least half the size of the biggest one.
    """
    mask = mask.astype(np.uint8)
    if size_threshold <= 0.0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:  # background + at most one component -> nothing to filter
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]  # row 0 is the background label
    cutoff = size_threshold * float(areas.max())
    out = np.zeros_like(mask)
    for label, area in enumerate(areas, start=1):
        if area >= cutoff:
            out[labels == label] = 1
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of every blob by re-filling its outer contour."""
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, contourIdx=-1, color=1, thickness=cv2.FILLED)
    return out


def dilate_mask(mask: np.ndarray, padding) -> np.ndarray:
    """Inflate a binary mask outward by `padding`.

    `padding` is an int (crop pixels) or a float (fraction of the mask's
    equivalent-circle radius, sqrt(area / pi)); 0 / 0.0 is a no-op. Dilation
    pushes blob boundaries outward, so polygon points extracted afterwards sit
    slightly outside the true edge rather than biting into the object.
    """
    mask = mask.astype(np.uint8)
    if isinstance(padding, float):
        area = int(np.count_nonzero(mask))
        radius = round(padding * (area / np.pi) ** 0.5)
    else:
        radius = int(padding)
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def _simplify(contour: np.ndarray, epsilon: float, max_points: int) -> np.ndarray:
    """Douglas-Peucker simplification, honouring `max_points`.

    epsilon < 1 is a fraction of the contour perimeter (resolution-independent);
    epsilon >= 1 is an absolute distance in crop pixels.
    """
    perimeter = cv2.arcLength(contour, True)
    base_eps = epsilon * perimeter if epsilon < 1.0 else float(epsilon)

    approx = cv2.approxPolyDP(contour, base_eps, True)
    if len(approx) <= max_points:
        return approx

    # Too many vertices: a larger epsilon yields fewer points (monotonic), so
    # binary-search upward for the smallest epsilon that fits under the cap.
    lo, hi = base_eps, max(base_eps, perimeter)
    best = approx
    for _ in range(_EPSILON_SEARCH_STEPS):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(contour, mid, True)
        if len(approx) <= max_points:
            best, hi = approx, mid
        else:
            lo = mid
    return best


def mask_to_polygons(mask: np.ndarray, epsilon: float, max_points: int) -> List[Polygon]:
    """Return the external contours of `mask` as simplified crop-coordinate polygons.

    Holes are ignored (RETR_EXTERNAL): a Label Studio polygon is a single ring.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[Polygon] = []
    for contour in contours:
        if cv2.contourArea(contour) < 1.0:
            continue
        approx = _simplify(contour, epsilon, max_points)
        points = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(points) >= 3:  # a polygon needs at least 3 vertices
            polygons.append(points)
    return polygons


def mask_to_rectangles(mask: np.ndarray):
    """Return connected-component bounding boxes as (x0, y0, x1, y1)."""
    mask = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for label in range(1, count):  # row 0 is background
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def draw_polygons(rgb: np.ndarray, polygons: List[Polygon],
                  line_thickness: int = 1, vertex_radius: int = 3) -> np.ndarray:
    """Return a copy of an RGB image with polygons drawn: thin green edges/dots."""
    out = np.ascontiguousarray(rgb.copy())
    green = (0, 190, 0)  # RGB
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=green, thickness=line_thickness)
    for polygon in polygons:  # draw vertices last so they sit on top of the edges
        for x, y in polygon:
            cv2.circle(out, (int(round(x)), int(round(y))), vertex_radius, green, -1)
    return out
