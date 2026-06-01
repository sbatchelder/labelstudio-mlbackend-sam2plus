#!/usr/bin/env python3
"""
probe.py -- submit one image + bounding box to the running SAM2_BigImg ML
backend and save the Label Studio prediction JSON.

The bounding box is four numbers, X Y W H, and the mode is chosen by syntax:

  * all written as integers (no '.')  -> PIXEL mode:
        top-left corner + width/height, in pixels
  * any written with a decimal point  -> RELATIVE mode:
        center point + width/height, as fractions of the image (0.0 - 1.0)

Mixing integers and decimals is rejected. A relative value of 0 or 1 must be
written "0.0" / "1.0", otherwise it reads as pixel mode.

How the image reaches the backend:
    The backend resolves images from its bind-mounted cache directory (it does
    not fetch arbitrary local paths). This script re-encodes the input to JPEG
    and copies it into that cache under a unique URL, so /predict resolves it
    fully offline -- no Label Studio server or API key required.

Outputs (into OUTDIR):
    prediction_rle.json / prediction_polygon.json   the raw /predict response
    bbox.jpg               full image with the prompt box drawn
    bbox.crop.jpg          the SAM2 input crop with the prompt box drawn
    crop.mask_rle.jpg      the predicted mask, in crop resolution
    crop.mask_polygon.jpg  the crop with the polygon overlaid (polygon mode only)
The backend writes further mask/crop side-effect images into the cache
directory; all their paths are printed so you can inspect them directly.

IMAGE and --bbox both default to COCO object 69 of the example core image, so
`python probe.py` with no arguments runs that case end to end.

--extra-args takes a JSON file of backend extra_params (CropMapper kwargs plus
optional `as_polygon` / `postprocess`); probe applies it via /setup before
/predict. With no --extra-args the backend behaves like the stock SAM2 model.
See examples/basic-rle.json and examples/basic-polygon.json.

Usage:
    python probe.py [IMAGE] [--bbox X Y W H] [-o OUTDIR] [--extra-args FILE]
                    [--url URL] [--cache-dir DIR] [--label LABEL]

Examples:
    python probe.py
    python probe.py examples/core.jpg --bbox 4600 4400 600 600 -o probe_out
    python probe.py examples/core.jpg --bbox 0.45 0.45 0.06 0.06 -o probe_out
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the backend targets very large images

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_URL = os.environ.get("ML_BACKEND_URL", "http://localhost:22202")
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_IMAGE = REPO_ROOT / "examples" / "117_U1553D_4R_2W_16-21cm_N1of1_Z200x.jpg"
DEFAULT_BBOX = ["3909", "2408", "234", "320"]  # COCO obj 69 of the default image

# Project id shared between /setup and /predict so the extra_params applied by
# /setup are read back by /predict (the backend keys them per project id; this
# relies on the backend running a single worker, as docker-compose configures).
PROJECT_ID = "probe"

# Side-effect images copied from the cache into OUTDIR: cache suffix -> output
# name. Missing sources are skipped, so the polygon overlay only lands in
# OUTDIR when the backend ran in polygon mode.
COPIED_SIDE_EFFECTS = {
    ".bbox.jpg": "bbox.jpg",
    ".crop.bbox.jpg": "bbox.crop.jpg",
    ".mask.crop.jpg": "crop.mask_rle.jpg",
    ".crop.mask_polygon.jpg": "crop.mask_polygon.jpg",
}

# predict() resolves either BrushLabels or PolygonLabels (by mode) against the
# Image tag; the smart RectangleLabels drives the interactive bbox prompt.
LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="defect"/>
  </BrushLabels>
  <PolygonLabels name="tag2" toName="image">
    <Label value="defect"/>
  </PolygonLabels>
  <RectangleLabels name="tag3" toName="image" smart="true">
    <Label value="defect"/>
  </RectangleLabels>
</View>
"""

_INT_RE = re.compile(r"^[+-]?\d+$")


def classify_bbox(raw):
    """Return ('pixel'|'relative', [float, float, float, float]) for 4 raw strings."""
    is_int = [bool(_INT_RE.match(v)) for v in raw]
    try:
        values = [float(v) for v in raw]
    except ValueError as exc:
        raise SystemExit(f"error: bbox values must be numbers ({exc})")

    if all(is_int):
        mode = "pixel"
    elif not any(is_int):
        mode = "relative"
    else:
        raise SystemExit(
            "error: bbox mixes integers and decimals. Use four integers for "
            "pixel mode or four decimals for relative mode (write 0.0 / 1.0)."
        )
    return mode, values


def to_ls_rectangle(mode, values, width, height):
    """Convert a pixel/relative bbox to a Label Studio rectangle (percent, top-left)."""
    if mode == "pixel":
        x, y, w, h = values
        return x / width * 100, y / height * 100, w / width * 100, h / height * 100

    # relative: center + size as fractions -> top-left percentages
    cx, cy, w, h = values
    for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not 0.0 <= val <= 1.0:
            print(f"warning: relative bbox {name}={val} is outside 0.0-1.0",
                  file=sys.stderr)
    return (cx - w / 2) * 100, (cy - h / 2) * 100, w * 100, h * 100


def load_extra_args(path):
    """Load a JSON object of backend extra_params; {} when no file is given."""
    if not path:
        return {}
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"error: extra-args file not found: {file}")
    try:
        data = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: {file} is not valid JSON ({exc})")
    if not isinstance(data, dict):
        raise SystemExit(
            f"error: {file} must contain a JSON object, got {type(data).__name__}"
        )
    return data


def seed_image_into_cache(image_path, cache_dir):
    """Re-encode the image to JPEG inside the backend cache; return its image URL."""
    stem = Path(image_path).stem
    # Unique per run so each probe gets its own cache entry and side-effect files.
    url = f"http://sam2bigimg.probe/{uuid4().hex}/{stem}.jpg"
    digest = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8]
    cache_name = f"{digest}__{os.path.basename(urlparse(url).path)}"
    dest = cache_dir / cache_name

    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        width, height = im.size
        im.convert("RGB").save(dest, "JPEG", quality=95)
    return url, dest, width, height


def main():
    parser = argparse.ArgumentParser(
        description="Submit an image + bbox to the SAM2_BigImg backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE),
                        help=f"path to the input image (default: {DEFAULT_IMAGE.name})")
    parser.add_argument("--bbox", nargs=4, default=DEFAULT_BBOX,
                        metavar=("X", "Y", "W", "H"),
                        help="bounding box: 4 ints (pixel TL+WH) or 4 decimals "
                             "(relative center+WH); default: "
                             f"{' '.join(DEFAULT_BBOX)}")
    parser.add_argument("-o", "--out", default="probe_out",
                        help="output directory for prediction.json (default: probe_out)")
    parser.add_argument("--extra-args", default=None,
                        help="JSON file of backend extra_params applied via "
                             "/setup; omit for stock (brush RLE) behaviour")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"ML backend base URL (default: {DEFAULT_URL})")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help="host directory bind-mounted to the backend's /cache "
                             f"(default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--label", default="defect",
                        help="brush label to assign (default: defect)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="HTTP timeout in seconds (default: 300)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"error: image not found: {image_path}")

    mode, values = classify_bbox(args.bbox)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser()
    base_url = args.url.rstrip("/")
    extra_args = load_extra_args(args.extra_args)
    as_polygon = bool(extra_args.get("as_polygon"))

    url, cached_image, width, height = seed_image_into_cache(image_path, cache_dir)
    x_pct, y_pct, w_pct, h_pct = to_ls_rectangle(mode, values, width, height)

    print(f"image      : {image_path}  ({width}x{height})")
    print(f"bbox mode  : {mode}  ->  x={x_pct:.3f}% y={y_pct:.3f}% "
          f"w={w_pct:.3f}% h={h_pct:.3f}%")
    print(f"extra args : {extra_args or '(none)'}")
    print(f"cached as  : {cached_image}")

    # Apply extra CropMapper kwargs through /setup; they are keyed to PROJECT_ID
    # and read back by /predict below (which carries the same project id).
    print(f"posting    : {base_url}/setup ...")
    try:
        setup_resp = requests.post(
            f"{base_url}/setup",
            json={"project": PROJECT_ID, "schema": LABEL_CONFIG,
                  "extra_params": json.dumps(extra_args)},
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"error: could not reach backend at {base_url}: {exc}")
    if setup_resp.status_code != 200:
        raise SystemExit(f"error: /setup returned {setup_resp.status_code}: {setup_resp.text}")

    request_body = {
        "project": PROJECT_ID,
        "tasks": [{"id": 1, "data": {"image": url}}],
        "label_config": LABEL_CONFIG,
        "params": {
            "context": {
                "result": [{
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "from_name": "tag3",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x_pct, "y": y_pct,
                        "width": w_pct, "height": h_pct,
                        "rectanglelabels": [args.label],
                    },
                }],
            },
        },
    }

    print(f"posting    : {base_url}/predict ...")
    try:
        resp = requests.post(f"{base_url}/predict", json=request_body,
                             timeout=args.timeout)
    except requests.RequestException as exc:
        raise SystemExit(f"error: could not reach backend at {base_url}: {exc}")
    if resp.status_code != 200:
        raise SystemExit(f"error: backend returned {resp.status_code}: {resp.text}")

    prediction = resp.json()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / ("prediction_polygon.json" if as_polygon
                           else "prediction_rle.json")
    json_path.write_text(json.dumps(prediction, indent=2))

    results = prediction.get("results", [])
    per_polygon = None  # per-polygon point counts, filled in polygon mode below
    if results and results[0].get("result"):
        regions = results[0]["result"]
        score = results[0].get("score")
        if as_polygon:
            per_polygon = [len(r["value"]["points"]) for r in regions]
            total_pts = sum(per_polygon)
            print(f"result     : score={score!r} polygons={len(regions)} "
                  f"points_per_polygon={per_polygon} total_points={total_pts}")
        else:
            region = regions[0]
            print(f"result     : score={score!r} "
                  f"rle_runs={len(region['value']['rle'])} "
                  f"label={region['value']['brushlabels']}")
    else:
        print("result     : no predictions returned "
              "(check that the bbox lies inside the image)")
    print(f"json saved : {json_path}")

    # Side-effect images the backend writes into the cache for this run.
    stem = str(cached_image)[: -len(".jpg")]
    print("cache side-effect images:")
    for suffix in (".bbox.jpg", ".crop.jpg", ".crop.bbox.jpg",
                   ".mask.crop.jpg", ".mask.jpg", ".crop.mask_polygon.jpg"):
        path = Path(stem + suffix)
        mark = "" if path.exists() else "  (missing)"
        print(f"  {path}{mark}")

    # Copy a subset of side-effect images into the output directory.
    print("copied into output directory:")
    for suffix, out_name in COPIED_SIDE_EFFECTS.items():
        src = Path(stem + suffix)
        dst = out_dir / out_name
        if src.exists():
            shutil.copyfile(src, dst)
            note = ""
            if out_name == "crop.mask_polygon.jpg" and per_polygon is not None:
                note = f"  ({sum(per_polygon)} points: {per_polygon})"
            print(f"  {dst}{note}")
        else:
            print(f"  {dst}  (skipped: source missing)")


if __name__ == "__main__":
    main()
