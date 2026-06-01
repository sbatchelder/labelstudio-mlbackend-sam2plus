#!/usr/bin/env python3
"""Submit one prompt to a running SAM2Plus backend."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from string import Template
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import requests
import yaml
from label_studio_sdk.converter import brush
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = os.environ.get("ML_BACKEND_URL", "http://localhost:22202")
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_IMAGE = REPO_ROOT / "examples" / "01-data" / "ichthyoliths.jpg"
DEFAULT_PROMPT = {
    "type": "RectangleLabel",
    "label": "probe",
    "geometry": [3909, 2408, 234, 320],
}
PROJECT_ID = "probe"

_INT_RE = re.compile(r"^[+-]?\d+$")

_FORMATS = {
    "brushlabel": ("BrushLabels", "brushlabels", "brushlabels", True),
    "brush": ("Brush", "brush", None, False),
    "polygonlabel": ("PolygonLabels", "polygonlabels", "polygonlabels", True),
    "polygon": ("Polygon", "polygon", None, False),
    "rectanglelabel": ("RectangleLabels", "rectanglelabels", "rectanglelabels", True),
    "rectangle": ("Rectangle", "rectangle", None, False),
    "keypointlabel": ("KeyPointLabels", "keypointlabels", "keypointlabels", True),
    "keypoint": ("KeyPoint", "keypoint", None, False),
}


def normalize_type(value):
    key = str(value).replace("Labels", "Label").lower()
    if key not in _FORMATS:
        raise SystemExit(f"error: unsupported tag type: {value}")
    tag, result_type, label_key, labeled = _FORMATS[key]
    return {
        "requested": value,
        "type": tag,
        "tag": tag,
        "result_type": result_type,
        "label_key": label_key,
        "labeled": labeled,
    }


def _load_yaml(path):
    if not path:
        return {}
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"error: config file not found: {file}")
    data = yaml.safe_load(file.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"error: {file} must contain a YAML mapping")
    return data


def _load_json_object(path):
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
        raise SystemExit(f"error: {file} must contain a JSON object")
    return data


def _parser():
    parser = argparse.ArgumentParser(
        description="Submit an image + prompt to a SAM2Plus backend.",
    )
    parser.add_argument("image", nargs="?", default=argparse.SUPPRESS)
    parser.add_argument("--config", default=None,
                        help="YAML config file; CLI options override config values")
    parser.add_argument("--prompt", default=argparse.SUPPRESS,
                        help="JSON prompt object; YAML configs are clearer")
    parser.add_argument("--extra-args", default=argparse.SUPPRESS,
                        help="JSON file of backend extra_params")
    parser.add_argument("--url", default=argparse.SUPPRESS,
                        help=f"ML backend base URL (default: {DEFAULT_URL})")
    parser.add_argument("--cache-dir", default=argparse.SUPPRESS,
                        help=f"host cache dir mounted to /cache (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--timeout", type=float, default=argparse.SUPPRESS,
                        help="HTTP timeout in seconds (default: 300)")
    parser.add_argument("--project", default=argparse.SUPPRESS,
                        help="Label Studio project id shared by /setup and /predict")
    parser.add_argument("--name", default=argparse.SUPPRESS,
                        help="request/artifact name")
    parser.add_argument("--request-record", nargs="?", const="probe_out/03-requests/${name}",
                        default=argparse.SUPPRESS, help="directory for POST request artifacts")
    parser.add_argument("--probe-fullframe-artifact-resize", type=float,
                        default=argparse.SUPPRESS,
                        help="optional scale for request/output full-frame image artifacts")
    parser.add_argument("--intermediates", nargs="?", const="probe_out/04-intermediates/${name}",
                        default=argparse.SUPPRESS, help="directory for intermediate patch graphics")
    parser.add_argument("--output", "-o", nargs="?", const="probe_out/05-outputs/${name}",
                        default=argparse.SUPPRESS, help="directory for prediction.json")
    parser.add_argument("--output-imgs", "--output-img", nargs="?",
                        const="probe_out/05-outputs/${name}", default=argparse.SUPPRESS,
                        help="directory for output image artifacts")
    return parser


def _merged_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    cfg = _load_yaml(known.config)
    parsed = vars(_parser().parse_args(argv))
    parsed.pop("config", None)
    if "prompt" in parsed and isinstance(parsed["prompt"], str):
        parsed["prompt"] = json.loads(parsed["prompt"])

    merged = {
        "image": str(DEFAULT_IMAGE),
        "prompt": dict(DEFAULT_PROMPT),
        "extra_args": None,
        "url": DEFAULT_URL,
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "timeout": 300.0,
        "project": None,
        "name": None,
        "request_record": None,
        "probe_fullframe_artifact_resize": None,
        "intermediates": None,
        "output": None,
        "output_imgs": None,
    }
    merged.update(cfg)
    merged.update(parsed)

    if not merged["name"]:
        image_stem = Path(merged["image"]).stem
        return_type = return_format(merged)["requested"].lower()
        merged["name"] = f"{image_stem}-{return_type}"
    if not merged["project"]:
        merged["project"] = merged["name"]

    for key in ("request_record", "intermediates", "output", "output_imgs"):
        value = merged.get(key)
        if isinstance(value, str):
            merged[key] = Template(value).safe_substitute(name=merged["name"])

    return argparse.Namespace(**merged)


def return_format(args_or_dict):
    if isinstance(args_or_dict, dict):
        extra_path = args_or_dict.get("extra_args")
    else:
        extra_path = args_or_dict.extra_args
    extra = _load_json_object(extra_path)
    cfg = extra.get("return_format") or {"type": "BrushLabel"}
    fmt = normalize_type(cfg.get("type", "BrushLabel"))
    return fmt


def classify_numbers(raw):
    raw = [str(v) for v in raw]
    is_int = [bool(_INT_RE.match(v)) for v in raw]
    values = [float(v) for v in raw]
    if all(is_int):
        return "pixel", values
    if not any(is_int):
        return "relative", values
    raise SystemExit("error: geometry mixes integers and decimals")


def rectangle_to_percent(geometry, width, height):
    mode, values = classify_numbers(geometry)
    if mode == "pixel":
        x, y, w, h = values
        return x / width * 100, y / height * 100, w / width * 100, h / height * 100
    cx, cy, w, h = values
    return (cx - w / 2) * 100, (cy - h / 2) * 100, w * 100, h * 100


def keypoint_to_percent(geometry, width, height):
    mode, values = classify_numbers(geometry)
    if len(values) != 2:
        raise SystemExit("error: keypoint geometry must have two values")
    x, y = values
    if mode == "pixel":
        return x / width * 100, y / height * 100
    return x * 100, y * 100


def rect_percent_to_pixels(rect, width, height):
    x, y, w, h = rect
    return (
        int(round(x * width / 100.0)),
        int(round(y * height / 100.0)),
        int(round((x + w) * width / 100.0)),
        int(round((y + h) * height / 100.0)),
    )


def point_percent_to_pixels(point, width, height):
    x, y = point
    return int(round(x * width / 100.0)), int(round(y * height / 100.0))


def seed_image_into_cache(image_path, cache_dir):
    stem = Path(image_path).stem
    url = f"http://sam2plus.probe/{uuid4().hex}/{stem}.jpg"
    digest = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8]
    cache_name = f"{digest}__{os.path.basename(urlparse(url).path)}"
    dest = cache_dir / cache_name

    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        width, height = im.size
        im.convert("RGB").save(dest, "JPEG", quality=95)
    return url, dest, width, height


def build_label_config(prompt, output):
    prompt_type = normalize_type(prompt["type"])
    output_type = output

    def tag_xml(tag_type, name, label=None, smart=False):
        smart_attr = ' smart="true"' if smart else ""
        if label and normalize_type(tag_type)["labeled"]:
            return (
                f'  <{normalize_type(tag_type)["tag"]} name="{name}" toName="image"{smart_attr}>\n'
                f'    <Label value="{label}"/>\n'
                f'  </{normalize_type(tag_type)["tag"]}>'
            )
        return f'  <{normalize_type(tag_type)["tag"]} name="{name}" toName="image"{smart_attr}/>'

    return "\n".join([
        "<View>",
        '  <Image name="image" value="$image" zoom="true"/>',
        tag_xml(prompt["type"], "tag1", prompt.get("label"), smart=True),
        tag_xml(output_type["requested"], "tag2", prompt.get("label"), smart=False),
        "</View>",
    ])


def build_context(prompt, width, height):
    prompt_type = normalize_type(prompt["type"])
    value = {}
    if prompt_type["result_type"] in {"rectanglelabels", "rectangle"}:
        x, y, w, h = rectangle_to_percent(prompt["geometry"], width, height)
        value.update({"x": x, "y": y, "width": w, "height": h})
    elif prompt_type["result_type"] in {"keypointlabels", "keypoint"}:
        x, y = keypoint_to_percent(prompt["geometry"], width, height)
        value.update({"x": x, "y": y})
    else:
        raise SystemExit("error: prompt.type must be RectangleLabel, Rectangle, KeyPointLabel, or KeyPoint")

    if prompt_type["label_key"]:
        value[prompt_type["label_key"]] = [prompt.get("label", "probe")]

    region = {
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "from_name": "tag1",
        "to_name": "image",
        "type": prompt_type["result_type"],
        "value": value,
    }
    if prompt_type["result_type"] in {"keypointlabels", "keypoint"}:
        region["is_positive"] = 1
    return {"result": [region]}


def prompt_pixels(prompt, width, height):
    prompt_type = normalize_type(prompt["type"])
    if prompt_type["result_type"] in {"rectanglelabels", "rectangle"}:
        return "rectangle", rect_percent_to_pixels(
            rectangle_to_percent(prompt["geometry"], width, height), width, height)
    return "keypoint", point_percent_to_pixels(
        keypoint_to_percent(prompt["geometry"], width, height), width, height)


def draw_prompt(image, prompt_shape):
    kind, geom = prompt_shape
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    if kind == "rectangle":
        draw.rectangle(geom, outline=(255, 0, 0), width=5)
    else:
        x, y = geom
        r = 12
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=5)
    return out


def polygon_pixels(region):
    width = region["original_width"]
    height = region["original_height"]
    return [
        (float(x) * width / 100.0, float(y) * height / 100.0)
        for x, y in region["value"]["points"]
    ]


def rectangle_pixels(region):
    width = region["original_width"]
    height = region["original_height"]
    value = region["value"]
    return rect_percent_to_pixels(
        (value["x"], value["y"], value["width"], value["height"]),
        width,
        height,
    )


def draw_polygon(draw, points, offset=(0, 0), color=(0, 180, 0), width=5, radius=7):
    if not points:
        return
    ox, oy = offset
    shifted = [(x - ox, y - oy) for x, y in points]
    draw.line(shifted + [shifted[0]], fill=color, width=width)
    for x, y in shifted:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def decode_rle_mask(region):
    width = region["original_width"]
    height = region["original_height"]
    decoded = brush.decode_rle(region["value"]["rle"])
    rgba = np.reshape(decoded, [height, width, 4])
    return rgba[:, :, 3] > 0


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clamp_box(box, image_size):
    left, top, right, bottom = box
    width, height = image_size
    return (
        max(0, left),
        max(0, top),
        min(width, right),
        min(height, bottom),
    )


def save_png_from_any(srcs, dst, transform=None):
    for src in srcs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                if transform:
                    im = transform(im)
                im.save(dst)
            return


def resize_image(image, scale):
    if not scale:
        return image
    size = (max(1, round(image.width * scale)),
            max(1, round(image.height * scale)))
    return image.resize(size)


def save_request_artifacts(out_dir, image_path, prompt_shape, extra_args,
                           predict_body, resize=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        input_image = draw_prompt(im, prompt_shape)
        input_image = resize_image(input_image, resize)
        input_image.save(out_dir / "input.jpg", "JPEG", quality=95)
    (out_dir / "POST-setup.json").write_text(json.dumps(extra_args, indent=2))
    (out_dir / "POST-predict.json").write_text(json.dumps(predict_body, indent=2))


def save_intermediates(out_dir, cached_image, output_format):
    out_dir = Path(out_dir)
    stem = str(cached_image)[: -len(".jpg")]
    save_png_from_any([Path(stem + ".patch.jpg")], out_dir / "patch.png")
    save_png_from_any([Path(stem + ".patch.bbox.jpg")], out_dir / "patch.prompt.png")
    if output_format["type"] in {"BrushLabels", "Brush"}:
        save_png_from_any([Path(stem + ".mask.patch.jpg")], out_dir / "patch.mask_rle.png")
    elif output_format["type"] in {"PolygonLabels", "Polygon"}:
        save_png_from_any([Path(stem + ".patch.mask_polygon.jpg")],
                          out_dir / "patch.mask_polygon.png")


def save_output_json(out_dir, prediction):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prediction.json").write_text(json.dumps(prediction, indent=2))


def save_output_images(out_dir, image_path, prediction, prompt_shape,
                       fullframe_resize=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = prediction.get("results", [])
    regions = results[0].get("result", []) if results else []
    if not regions:
        return

    with Image.open(image_path) as im:
        base = im.convert("RGB")
        full = draw_prompt(base, prompt_shape)
        draw = ImageDraw.Draw(full, "RGBA")
        first_type = regions[0]["type"]

        if first_type in {"polygonlabels", "polygon"}:
            all_points = []
            for region in regions:
                pts = polygon_pixels(region)
                all_points.extend(pts)
                draw_polygon(draw, pts, color=(0, 190, 0), width=1, radius=3)
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            pred_box = clamp_box((int(np.floor(min(xs))), int(np.floor(min(ys))),
                                  int(np.ceil(max(xs))) + 1, int(np.ceil(max(ys))) + 1),
                                 base.size)
            crop = base.crop(pred_box)
            crop.save(out_dir / "prediction_crop.png")
            annotated = crop.copy()
            crop_draw = ImageDraw.Draw(annotated, "RGBA")
            for region in regions:
                draw_polygon(crop_draw, polygon_pixels(region), offset=pred_box[:2],
                             color=(0, 190, 0), width=1, radius=3)
            annotated.save(out_dir / "prediction_crop_annotated.jpg", "JPEG", quality=95)
        elif first_type in {"rectanglelabels", "rectangle"}:
            boxes = []
            for region in regions:
                box = rectangle_pixels(region)
                boxes.append(box)
                draw.rectangle(box, outline=(0, 190, 0), width=1)
            pred_box = clamp_box((
                min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes),
            ), base.size)
            crop = base.crop(pred_box)
            crop.save(out_dir / "prediction_crop.png")
            annotated = crop.copy()
            crop_draw = ImageDraw.Draw(annotated, "RGBA")
            for box in boxes:
                offset_box = (box[0] - pred_box[0], box[1] - pred_box[1],
                              box[2] - pred_box[0], box[3] - pred_box[1])
                crop_draw.rectangle(offset_box, outline=(0, 190, 0), width=1)
            annotated.save(out_dir / "prediction_crop_annotated.jpg", "JPEG", quality=95)
        else:
            combined = np.zeros((base.height, base.width), dtype=bool)
            for region in regions:
                combined |= decode_rle_mask(region)
            mask_box = bbox_from_mask(combined)
            if mask_box is None:
                return
            alpha = Image.fromarray((combined.astype(np.uint8) * 110), "L")
            green = Image.new("RGBA", base.size, (0, 190, 0, 0))
            green.putalpha(alpha)
            full = Image.alpha_composite(full.convert("RGBA"), green).convert("RGB")
            pred_box = clamp_box(mask_box, base.size)
            base.crop(pred_box).save(out_dir / "prediction_crop.png")
            mask_crop = combined[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2]]
            Image.fromarray(mask_crop.astype(np.uint8) * 255, "L").save(
                out_dir / "prediction_crop_annotated.jpg",
                "JPEG",
                quality=95,
            )

        full = resize_image(full, fullframe_resize)
        full.save(out_dir / "fullframe_prediction.jpg", "JPEG", quality=95)


def build_request(project, image_url, width, height, context, label_config):
    return {
        "project": project,
        "tasks": [{"id": 1, "data": {"image": image_url}}],
        "label_config": label_config,
        "params": {"context": context},
    }


def main(argv=None):
    args = _merged_args(argv)
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"error: image not found: {image_path}")

    extra_args = _load_json_object(args.extra_args)
    output_format = return_format(args)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    base_url = args.url.rstrip("/")

    image_url, cached_image, width, height = seed_image_into_cache(image_path, cache_dir)
    label_config = build_label_config(args.prompt, output_format)
    context = build_context(args.prompt, width, height)
    prompt_shape = prompt_pixels(args.prompt, width, height)
    setup_body = {
        "project": args.project,
        "schema": label_config,
        "extra_params": json.dumps(extra_args),
    }
    predict_body = build_request(args.project, image_url, width, height,
                                 context, label_config)

    setup_resp = requests.post(f"{base_url}/setup", json=setup_body,
                               timeout=args.timeout)
    if setup_resp.status_code != 200:
        raise SystemExit(f"error: /setup returned {setup_resp.status_code}: {setup_resp.text}")

    resp = requests.post(f"{base_url}/predict", json=predict_body,
                         timeout=args.timeout)
    if resp.status_code != 200:
        raise SystemExit(f"error: backend returned {resp.status_code}: {resp.text}")

    prediction = resp.json()
    print(json.dumps(prediction, indent=2))

    if args.request_record:
        save_request_artifacts(args.request_record, image_path, prompt_shape,
                               extra_args, predict_body,
                               args.probe_fullframe_artifact_resize)
    if args.intermediates:
        save_intermediates(args.intermediates, cached_image, output_format)
    if args.output:
        save_output_json(args.output, prediction)
    if args.output_imgs:
        save_output_images(args.output_imgs, image_path, prediction, prompt_shape,
                           args.probe_fullframe_artifact_resize)


if __name__ == "__main__":
    main()
