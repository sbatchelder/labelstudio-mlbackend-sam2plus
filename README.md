<!--
---
title: SAM2 with Images
type: guide
tier: all
order: 15
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM2 with Label Studio for Image Annotation
categories:
    - Computer Vision
    - Image Annotation
    - Object Detection
    - Segment Anything Model
image: "/guide/ml_tutorials/sam2-images.png"
---
-->

# SAM2Plus

This is a modification of the [Labelstudio SAM2 ml-backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image), an ml-backend used to interactively make annotations on Labelstudio. 

This fork enhances the stock SAM2 ml-backend in several ways:

1. enables server-side mask post-process cleanup operations ([post-processing](#post-processing-postprocess))
2. enables compatibility with annotation types beyond BrushLabels ([return-format](#return-format-return-format))
3. enables efficient inferencing on small objects in very large input images ([subpatching](#subpatching-subpatching))
4. adds a compose configuration streamlining deployment of this ml-backend for multiple labelstudio instances ([multi-target deployment](#multi-target))
5. allows local usage of the ml-backend without a labelstudio instance (eg for testing) ([probe](#testing-with-sam2plus-probe))

## Installation

SAM2Plus runs as a Docker Compose service that Label Studio connects to as an ML
backend. For how Label Studio attaches to a backend from its side, see the Label
Studio guide on
[setting up an example ML backend](https://labelstud.io/guide/ml#Set-up-an-example-ML-backend).

1. **Clone the repository.**

   ```bash
   git clone https://github.com/sbatchelder/labelstudio-mlbackend-SAM2Plus.git
   cd labelstudio-mlbackend-SAM2Plus
   ```

2. **Provide SAM2 checkpoints.** Populate `data/model-store/` with the SAM2
   checkpoints before the first run, or point `MODEL_STORE` (in `envs/BASE.env`
   or a target env) at an already-populated checkpoint directory. It is mounted
   read-only at `/sam2/checkpoints`; an empty dir here shadows the checkpoints
   baked into the image.

3. **Create a target env file.** Each Label Studio instance the backend serves
   gets its own `envs/<target>.env`. Copy the template and edit it:

   ```bash
   cp envs/example.env envs/mytarget.env
   ```

   Set at least `CONTAINER_NAME`, `HOST_PORT`, `SERVER_DIR`, `CACHE_DIR`,
   `LABEL_STUDIO_URL`, and `LABEL_STUDIO_API_KEY`. Shared defaults (image name,
   device, model config, workers) live in `envs/BASE.env`; the target file
   overrides whatever it sets. API-key values are redacted on stage by the
   `envsecrets` filter — see [Secret Redaction](#secret-redaction).

4. **Launch the backend** with `envs/BASE.env` first and the target env second:

   ```bash
   docker compose -f compose.yml --env-file envs/BASE.env --env-file envs/mytarget.env up -d --build
   ```

   See [Deployment](#deployment) below for single- vs multi-instance variants.

To run SAM2Plus **without** a Label Studio server — for development or testing —
you only need a local virtualenv install; see
[Testing with `sam2plus-probe`](#testing-with-sam2plus-probe).

> SAM2Plus is a fork of HumanSignal's stock SAM2 example. The bootstrap notes for
> re-creating the fork live in
> [forking-original-sam2-mlbackend.md](forking-original-sam2-mlbackend.md).

### Deployment

Build and run with Docker Compose. The project has two compose entry points:

- `compose.yml` runs one backend instance using `envs/BASE.env` plus one target env.
- `multi-compose.yml` runs the non-local targets together (`brick` and `ichthyolith`) from one shared image.

Validate a running backend (localhost instance port shown):

```bash
curl http://localhost:22201/health
```

#### Single instance

A local/probe instance:

```bash
docker compose -f compose.yml --env-file envs/BASE.env --env-file envs/localhost.env up -d --build
```

A production target:

```bash
docker compose -f compose.yml --env-file envs/BASE.env --env-file envs/ichthyolith.env up -d --build
docker compose -f compose.yml --env-file envs/BASE.env --env-file envs/brick.env up -d --build
```

#### Multi-target

Run the non-local targets together from one shared image:

```bash
docker compose -f multi-compose.yml --env-file envs/BASE.env up -d --build sam2-build brick ichthyolith
```

`sam2-build` builds the shared image and exits cleanly; `brick` and
`ichthyolith` reference that image with their own ports, cache volumes, and
Label Studio settings.


## Labeling configuration

SAM2Plus runs as an **interactive** backend: the annotator places a prompt and
the model returns a region on the fly, rather than pre-labeling whole tasks in
the background. For background on this mode, see the Label Studio guide on
[interactive pre-annotations](https://labelstud.io/guide/ml#Interactive-pre-annotations).

The annotator draws a *prompt* (input modality) and SAM2 returns a *result*
(output modality).

Supported **input** prompt control tags:
- `KeyPointLabels` / `KeyPoint` — one or more click points
- `RectangleLabels` / `Rectangle` — a bounding box

Supported **output** result control tags (selected via
`return_format.type`, see [Return format](#return-format-return-format)):
- `BrushLabels` / `Brush` — RLE pixel mask (default)
- `PolygonLabels` / `Polygon` — simplified polygon ring(s)
- `RectangleLabels` / `Rectangle` — axis-aligned bounding box(es)

> **The prompt tag must set `smart="true"`.** That attribute is what turns the
> control into a per-region [smart tool](https://labelstud.io/guide/ml.html#Smart-tools)
> that triggers interactive prediction; without it the backend is never called.
> Only the prompt (input) tag needs it — not the output tag.

Your labeling configuration needs a smart input tag for the prompt you want to
use **and** the output control tag that matches the configured `return_format`.
The `*Labels` variants carry a label value; the bare `Brush` / `Polygon` /
`Rectangle` variants do not.

The minimal config below drives SAM2 with a **Rectangle** prompt and returns a
**Polygon** result (set `return_format.type` to `Polygon`, see
[Model parameters](#model-parameters-extra_params)):

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Rectangle name="prompt" toName="image" smart="true"/>
  <Polygon name="output" toName="image"/>
</View>
```

Swap the tags to match other modalities — for example a `KeyPoint` prompt with a
`Brush` result — keeping `smart="true"` on the prompt tag only, and setting
`return_format.type` to match the output tag.


## Configuration
Parameters are split across env files:

- `envs/BASE.env` — shared defaults for both compose files, plus the shared image name.
- `envs/example.env` — template for a new target instance; copy it (see [Installation](#installation)).
- `envs/localhost.env`, `envs/brick.env`, `envs/ichthyolith.env` — target-specific ports, container names, cache dirs, GPU, and Label Studio URL/API key.

The common base parameters are:
- `DEVICE` - compute device for the model server (`cuda` by default; `cpu` is selectable but untested/much slower)
- `MODEL_CONFIG` - SAM2 model configuration file (`configs/sam2.1/sam2.1_hiera_l.yaml` by default)
- `MODEL_CHECKPOINT` - SAM2 model checkpoint file (`sam2.1_hiera_large.pt` by default)
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server
- `HOST_PORT` - host port that maps to container port `9090` in single-instance mode
- `HOST_GPU` - GPU device id exposed to the container
- `SERVER_DIR` - host directory mounted at `/data`
- `CACHE_DIR` - host directory mounted at `/cache`
- `MODEL_STORE` - host checkpoint directory mounted read-only at `/sam2/checkpoints`

### Secret Redaction

Env files are marked in `.gitattributes` for the `envsecrets` Git clean
filter. Configure it once per clone before staging env files:

```bash
git config filter.envsecrets.clean 'python scripts/clean-env-secrets.py'
git config filter.envsecrets.smudge cat
```

After that, staged env-file content redacts `LABEL_STUDIO_API_KEY`,
`LABEL_STUDIO_ACCESS_TOKEN`, and `BASIC_AUTH_PASS` values. The working tree
keeps the real values.


## Model parameters (`extra_params`)

When you set up a Label Studio project's ML backend (Model), you can supply
`extra_params`. SAM2Plus exposes a number of `extra_params` options, and they are
the primary means of configuring it. If `extra_params` is left blank for a
project's Model, SAM2Plus behaves exactly like the original example
[SAM2 ml-backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image)
provided by [HumanSignal](https://github.com/HumanSignal/label-studio-ml-backend):
the full image is sent to SAM2 and a `BrushLabels` RLE mask is returned.

Each top-level `extra_params` key enables a behavior independently:
`fullframe_resize` resizes the input before SAM, `subpatching` crops around the
prompt, `postprocess` edits the predicted mask before output conversion, and
`return_format` changes the returned Label Studio geometry. Output annotations
are always mapped back to the original full-resolution image coordinates.

`extra_params` are static per-project values. Label Studio sends them to the
backend's `/setup` endpoint; for local testing `sam2plus-probe` does the same via
`--extra-params` (see below). Unknown or misplaced keys raise an error instead of
being ignored.

> **Key naming.** Keys may use either underscores or dashes — both are valid;
> dashes are normalized to underscores internally (so `patch-size` and
> `patch_size` are the same key, and mixing the two forms of the same key in one
> object is an error). To avoid redundancy, the rest of this document writes
> multi-word keys in **dash** form only.

> **SAM2 always works at 1024².** Every SAM2 hiera config (`sam2_hiera_*` and
> `sam2.1_hiera_*`, all of `t`/`s`/`b+`/`l`) sets `image_size: 1024`, so SAM2
> internally resizes whatever image it is handed to 1024×1024 before encoding.
> `fullframe-resize` and `subpatching.patch-size` change **what** is fed to SAM2
> (and therefore the effective resolution and aspect ratio of the object inside
> that 1024² box) — they do not bypass this final resize. The whole point of
> `subpatching` is to make the object fill more of that 1024² so small objects in
> large scenes keep their detail.

### Full-frame resize (`fullframe-resize`)

If `fullframe-resize` is omitted, SAM2 receives the original full-resolution
image unless `subpatching` is configured. If supplied, the full frame is resized
before prompt coordinates, patching, and SAM inference. The value may be:

| Value | Meaning |
|-------|---------|
| float `> 0` and `<= 1` | scale both dimensions by this fraction, for example `0.5` halves width and height |
| int | resize to an exact square `size x size` |
| `[width, height]` | resize to exact dimensions |

Only downscaling is allowed. Masks, polygons, and rectangles are scaled back to
the original image size before they are returned to Label Studio.

Note this resize is independent of SAM2's own internal resize: SAM2 always
rescales its input to the 1024² of its hiera config regardless of
`fullframe-resize`. Downscaling the full frame mainly trades detail for speed and
memory; to *preserve* small-object detail, use `subpatching` instead.

### Return format (`return-format`)

Set `return-format.type` to choose the Label Studio result geometry:
`BrushLabels` (default), `Brush`, `PolygonLabels`, `Polygon`,
`RectangleLabels`, or `Rectangle`. The `*Labels` variants include the
configured label value in the returned region; the unlabeled variants do not.
This field is case-insensitive, so `brushlabels` is accepted. Singular
`BrushLabel` / `PolygonLabel` / `RectangleLabel` forms are invalid.

Polygon formats accept these additional `return-format` keys:

| Key | Default | Meaning |
|-----|---------|---------|
| `epsilon` | `1` | [Douglas–Peucker simplification](https://cartography-playground.gitlab.io/playgrounds/douglas-peucker-algorithm/) tolerance. `>= 1` is an absolute distance in crop pixels. `< 1` is treated as a fraction of the contour perimeter, so useful values are usually small, for example `0.003`. Larger = fewer, coarser vertices. |
| `max-points` | `100` | hard cap on vertices per polygon; `epsilon` is raised by a binary search until the cap is met |

The selected output requires a matching control tag in the labeling config:
`<BrushLabels>`, `<Brush>`, `<PolygonLabels>`, `<Polygon>`,
`<RectangleLabels>`, or `<Rectangle>`. Polygon formats implicitly require
filled holes — see below.

### Post-processing (`postprocess`)

If `postprocess` is omitted, the mask is not post-processed, except polygon
formats implicitly run with `fill-holes=true` because a polygon ring cannot
represent holes. If `postprocess.fill-holes` is explicitly set to `false` for
`PolygonLabels` or `Polygon`, the request errors.

When enabled, post-processing is applied to the binary mask before it is turned
into a brush mask, polygon, or rectangle:

| Key | Default | Meaning |
|-----|---------------------------|---------|
| `mask-size-threshold` | `0` | keep connected components whose area ≥ threshold × the largest component. `1` = largest blob only; `0.5` = blobs at least half its size; `0` = keep everything |
| `fill-holes` | `false` (`true` for polygons) | fill the interior holes of each blob. Must not be `false` for polygon return formats |
| `dilate` | `0` | inflate the mask outward (morphological dilation) so polygon points/edges sit just outside the true boundary instead of biting into the object. An **int** is an absolute distance in crop pixels; a **float** is a fraction of the mask's equivalent-circle radius `sqrt(area / π)` (scales with object size). `0` / `0.0` = no-op |

With polygon or rectangle return formats and `mask-size-threshold < 1`,
multiple surviving blobs produce multiple regions.

Post-processing runs in order: `mask-size-threshold` → `fill-holes` →
`dilate`, so noise blobs are dropped before the mask is inflated.

### Subpatching (`subpatching`)

Subpatching produces high-quality masks even on very large input images. When an
image arrives, SAM2Plus crops a `patch-size` patch around the annotator's
point/bounding-box prompt, runs SAM2 on that patch, and remaps the resulting mask
back to the original image's coordinates before returning it to Label Studio.

Without this, small objects in large scenes lose definition, because SAM2
internally resizes every input to a fixed 1024×1024; cropping first keeps the
object large within that view, preserving mask resolution. At the default
`patch-size` of `1024` the crop is fed to SAM2 at essentially 1:1, so an object
that fits the patch is segmented at its native resolution.

In the case where a prompt is larger than the `patch-size`, by default the `patch-size` increases to accommodate the larger prompt (SAM2 input no longer native resolution, but likely still better than fullframe input). `allow-oversize` and `oversize-padding` control this behavior. If the patch would have to grow past the (post-`fullframe-resize`) frame in either dimension to fit the prompt, subpatching is skipped for the given input prompt and the backend falls back to full-frame inference. If an input fullframe image is smaller than the patch-size, padding is applied around the fullframe image.

| Key | Default | Meaning |
|-----|---------|---------|
| `patch-size` | `1024` | patch edge length in px — an int (square) or `[width, height]` |
| `mode` | `"default"` | `default` keeps the patch inside the image bounds; `padding` centers it on the prompt and pads past edges |
| `padding-fill` | `"black"` | fill color used by `padding` mode |
| `allow-oversize` | `true` | grow the patch when the prompt box is larger than `patch-size`; if `false`, oversized prompts return an error |
| `oversize-padding` | `0.05` | extra fractional margin added when the patch is grown to fit an oversized box |

`patch-size` sets how large a region is cropped around the prompt; it does **not**
sidestep SAM2's internal resize. The cropped patch is still rescaled to the
1024² of the hiera config before encoding, so a 1024 `patch-size` feeds SAM2 a
near-1:1 crop while a much larger `patch-size` will be downscaled again inside SAM2.

### Example

```json
{
  "fullframe_resize": 0.5,
  "subpatching": { "patch_size": 1024 },
  "postprocess": { "mask_size_threshold": 1, "fill_holes": true, "dilate": 0 },
  "return_format": { "type": "PolygonLabels", "epsilon": 0.003, "max_points": 100 }
}
```

(JSON keys above use the underscore form to match the shipped example files;
dash form works identically.)

Ready-made files: `examples/02-extra_params/basic-rle.json` (patch + brush) and
`examples/02-extra_params/basic-polygon.json` (patch + polygon).


## Testing with `sam2plus-probe`

SAM2Plus ships a small command-line client, `sam2plus-probe`, so you can exercise
the backend **without a Label Studio server** — handy for development, for tuning
`extra_params`, and for reviewing exactly what SAM2 saw and produced. It talks to
a running backend container (for local work, the `localhost` instance from
[Deployment](#deployment)), so the only local install you need is the probe
package itself in a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

`sam2plus-probe` submits one image and one prompt (bounding box or keypoint) to a
running backend — no Label Studio server required. It re-encodes the image into
the backend's bind-mounted cache so `/predict` resolves it offline, which makes
the probe the primary way to **test and review the ML backend offline**: drive a
real `/setup` + `/predict` cycle, then inspect exactly what the backend saw and
produced. By default it writes only the prediction JSON to stdout.

**Backend caching and side-effect files.** Every image the backend resolves is
cached under the container's `/cache` (the host `CACHE_DIR`). When `subpatching`
is active the backend also writes its intermediates next to the cached image —
the cropped patch (`*.patch.jpg`), the patch with the prompt drawn on it
(`*.patch.bbox.jpg`), the full-frame image with the prompt box (`*.bbox.jpg`),
and the resulting mask (`*.mask.patch.jpg` / `*.mask.jpg`, or
`*.patch.mask_polygon.jpg` for polygon output). These persist between runs, so a
prior run's artifacts may linger; `sam2plus-probe --intermediates` simply
collects the current run's copies into a tidy folder for review.

```bash
sam2plus-probe [IMAGE] [--prompt JSON] [--extra-params FILE] [--url URL]
```

- `IMAGE` (positional) and the prompt both default to a bundled example —
  `examples/01-data/ichthyoliths.jpg` with a rectangle prompt — so
  `sam2plus-probe` with no arguments runs end to end.
- `--prompt JSON` — a JSON prompt object, for example
  `{"type": "RectangleLabel", "label": "probe", "geometry": [3909, 2408, 234, 320]}`.
  `type` may be `RectangleLabel`/`Rectangle` (`geometry` = `[x, y, w, h]`) or
  `KeyPointLabel`/`KeyPoint` (`geometry` = `[x, y]`). Within a geometry, all
  **integers** are read as pixels (`top-left + width/height` for a box); all
  **decimals** as relative coordinates (`center + width/height`, fractions of the
  image). A YAML `--config` is usually clearer than inline JSON.
- `--extra-params FILE` — a JSON file of `extra_params` (see above), applied via
  `/setup` before `/predict`. Omit it for stock (full-image brush) behavior.
- `--url` — backend base URL (default `http://localhost:22201`).
- `--config FILE` — YAML config containing the same option names (`image`,
  `prompt`, `extra_params`, `name`, output dirs, …); CLI flags override it.
- `--probe-fullframe-artifact-resize SCALE` — resize full-frame probe artifacts
  such as `input.jpg` and `fullframe_prediction.jpg`.

Artifact output is opt-in. The bundled probe configs exercise the three return
formats and write their artifacts under `examples/`:

```bash
sam2plus-probe --config examples/00-probe_configs/ichthyo-rle.yaml
sam2plus-probe --config examples/00-probe_configs/ichthyo-polygon.yaml
sam2plus-probe --config examples/00-probe_configs/ichthyo-keypoint-rle.yaml
```

The `--request-record`, `--intermediates`, `--output` / `-o`, and
`--output-imgs` flags each take an optional directory (default
`probe_out/...`; the bundled configs above redirect them under `examples/`).
`${name}` in a path expands to the request name. They populate, respectively:

- `…/03-requests/<request-name>/` — input image with red prompt geometry and request JSON.
- `…/04-intermediates/<request-name>/` — backend patch/mask intermediate graphics.
- `…/05-outputs/<request-name>/` — returned `prediction.json`, full-frame prompt/prediction overlay, and tight prediction crops.

> `sam2plus-probe` applies `extra_params` with a `/setup` call and reads them back in
> the following `/predict` call, keyed by a shared project id. This only works
> when the backend runs a **single worker** (`WORKERS=1`, as `BASE.env` sets).

Automated API tests live under `tests/`:

```bash
python -m pytest tests/regular -v
python -m pytest tests/bigimg -v
python -m pytest tests -v
```
