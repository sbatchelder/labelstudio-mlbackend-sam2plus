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

This is a modification of the [Labelstudio SAM2 ml-backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image). It can be used to interactively add mask (brush) annotations on Labelstudio. 

This fork allows for the creation of masks for small objects within a very large input image.
The when an image is received by the ml-endpoint, an area around the Labelstudio annotator's point/bounding-box prompt is cropped from the original image. The SAM2 mask is generated using the crop, and the resulting mask's coordinates are then remapped to the original image's coordinates and returned to Labelstudio. 
Without this, masks for small objects in large scenes loose definition, since SAM2 resizes input images to WxH resolution. With this method, mask resolution is preserved. 

## TODOs
- ~~customize patch area (pixel size, fraction-increase of input bbox)~~ — done: `subpatching.patch_size` / `subpatching.oversize_padding` (see [Model parameters](#model-parameters-extra_params))
- mask cleanup: ~~single-mask, fill gaps, grow edges~~ done via `postprocess`; smooth edges still open
- ~~allow multiple labelstudio instances to use endpoint~~ — done via `multi-compose.yml`
- ~~Labelstudio Host, API Key, host-port, target GPU as .env variables~~ — done via `envs/`
- ~~easy endpoint testing from within project but outside running container~~ - done: `sam2plus-probe`

## Deployment

Build and run with Docker Compose. The project has two compose entry points:

- `compose.yml` runs one backend instance using `envs/SINGLE.env` plus one target env.
- `multi-compose.yml` runs the non-local targets together (`brick` and `ichthyolith`) from one shared image.

Populate `data/model-store/` with SAM2 checkpoints before first run, or set
`MODEL_STORE` in the env file to another populated checkpoint directory.

Single local/probe instance:

```bash
docker compose -f compose.yml --env-file envs/SINGLE.env --env-file envs/localhost.env up -d --build
```

Single production target:

```bash
docker compose -f compose.yml --env-file envs/SINGLE.env --env-file envs/ichthyolith.env up -d --build
docker compose -f compose.yml --env-file envs/SINGLE.env --env-file envs/brick.env up -d --build
```

Multi-target run:

```bash
docker compose -f multi-compose.yml --env-file envs/MULTI.env up -d --build sam2-build brick ichthyolith
```

`sam2-build` builds the shared image and exits cleanly; `brick` and
`ichthyolith` reference that image with their own ports, cache volumes, and
Label Studio settings.

Validate a running backend:

```bash
curl http://localhost:22202/health
```

## Initial Project Setup Notes

```
mkdir SAM2Plus && cd SAM2Plus
git init
git remote add upstream https://github.com/HumanSignal/label-studio-ml-backend.git
git fetch upstream master
git archive upstream/master label_studio_ml/examples/segment_anything_2_image | tar -x --strip-components=3
git add -A && git commit -m "init sam2 ml-backend"
```
Then update `model.py`, incl. renaming for `class NewModel` to `class SAM2Plus` and subsequent changes in `_wsgi.py`. 


## Labeling configuration

The current implementation of the Label Studio SAM2 ML backend works using Interactive mode. The user-guided inputs are:
- `KeypointLabels`
- `RectangleLabels`

And then SAM2 outputs `BrushLabels` as a result.

This means all three control tags should be represented in your labeling configuration:

```xml
<View>
<Style>
  .main {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 20px;
  }
  .container {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
  }
  .column {
    flex: 1;
    padding: 10px;
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    text-align: center;
  }
  .column .title {
    margin: 0;
    color: #333;
  }
  .column .label {
    margin-top: 10px;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 3px;
  }
  .image-container {
    width: 100%;
    height: 300px;
    background-color: #ddd;
    border-radius: 5px;
  }
</Style>
<View className="main">
  <View className="container">
    <View className="column">
      <View className="title">Choose Label</View>
      <View className="label">
        <BrushLabels name="tag" toName="image">
          
          
        <Label value="defect" background="#FFA39E"/></BrushLabels>
      </View>
    </View>
    <View className="column">
      <View className="title">Use Keypoint</View>
      <View className="label">
        <KeyPointLabels name="tag2" toName="image" smart="true">
          
          
        <Label value="defect" background="#250dd3"/></KeyPointLabels>
      </View>
    </View>
    <View className="column">
      <View className="title">Use Rectangle</View>
      <View className="label">
        <RectangleLabels name="tag3" toName="image" smart="true">
          
          
        <Label value="defect" background="#FFC069"/></RectangleLabels>
      </View>
    </View>
  </View>
  <View className="image-container">
    <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  </View>
</View>
</View>
```


## Configuration
Parameters are split across env files:

- `envs/SINGLE.env` — defaults for one `compose.yml` instance.
- `envs/MULTI.env` — defaults for `multi-compose.yml` and the shared image name.
- `envs/localhost.env`, `envs/brick.env`, `envs/ichthyolith.env` — target-specific ports, container names, cache dirs, GPU, and Label Studio URL/API key.


The following common parameters are available:
- `DEVICE` - specify the device for the model server (currently only `cuda` is supported, `cpu` is coming soon)
- `MODEL_CONFIG` - SAM2 model configuration file (`sam2_hiera_l.yaml` by default)
- `MODEL_CHECKPOINT` - SAM2 model checkpoint file (`sam2_hiera_large.pt` by default)
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
git config filter.envsecrets.clean '.venv/bin/python scripts/clean-env-secrets.py'
git config filter.envsecrets.smudge cat
```

After that, staged env-file content redacts `LABEL_STUDIO_API_KEY`,
`LABEL_STUDIO_ACCESS_TOKEN`, and `BASIC_AUTH_PASS` values. The working tree
keeps the real values.


## Model parameters (`extra_params`)

With **no** `extra_params` the backend behaves exactly like the stock
[SAM2 image backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image):
the full image is sent to SAM2 and a brush (RLE) mask is returned. Supplying
**any** `extra_params` switches on the enhanced pipeline - patch around the prompt,
optionally post-process, and optionally return polygons instead of brush masks.

`extra_params` are static per-project values. Label Studio sends them to the
backend's `/setup` endpoint; for local testing `sam2plus-probe` does the same via
`--extra-params` (see below). Unknown keys are logged as warnings and ignored.

### Subpatching (`subpatching`)

Patch options live under the top-level `subpatching` object and are forwarded to
`CropMapper` (`seg_cropper.py`):

| Key | Default | Meaning |
|-----|---------|---------|
| `patch_size` | `1024` | patch edge length in px — an int (square) or `[width, height]` |
| `mode` | `"default"` | `default` keeps the patch inside the image bounds; `padding` centers it on the prompt and pads past edges |
| `padding_fill` | `"black"` | fill color used by `padding` mode |
| `allow_size_override` | `true` | grow the patch when the prompt box is larger than `patch_size` |
| `oversize_padding` | `0.05` | extra fractional margin added when the patch is grown to fit an oversized box |

### Return format (`return_format`)

Set `return_format.type` to choose the Label Studio result geometry:
`BrushLabel` (default), `Brush`, `PolygonLabel`, `Polygon`, `RectangleLabel`,
or `Rectangle`. The `*Label` variants include the configured label value in
the returned region; the unlabeled variants do not.

Polygon formats accept these additional `return_format` keys:

| Key | Default | Meaning |
|-----|---------|---------|
| `epsilon` | `1` | Douglas–Peucker simplification tolerance. `< 1` is treated as a fraction of the contour perimeter (resolution-independent); `>= 1` is an absolute distance in crop pixels. Larger = fewer, coarser vertices. |
| `max_points` | `100` | hard cap on vertices per polygon; `epsilon` is raised by a binary search until the cap is met |

The selected output requires a matching control tag in the labeling config:
`<BrushLabels>`, `<Brush>`, `<PolygonLabels>`, `<Polygon>`,
`<RectangleLabels>`, or `<Rectangle>`. Polygon and rectangle formats shift the
`postprocess` defaults — see below.

### Post-processing (`postprocess`)

Applied to the binary mask before it is turned into a brush mask or polygons:

| Key | Default (brush / polygon) | Meaning |
|-----|---------------------------|---------|
| `mask_size_threshold` | `0` / `1` | keep connected components whose area ≥ threshold × the largest component. `1` = largest blob only; `0.5` = blobs at least half its size; `0` = keep everything |
| `fill_holes` | `false` / `true` | fill the interior holes of each blob. Must be `true` for polygon return formats (a polygon ring cannot encode a hole) — otherwise the request errors |
| `dilate` | `0` | inflate the mask outward (morphological dilation) so polygon points/edges sit just outside the true boundary instead of biting into the object. An **int** is an absolute distance in crop pixels; a **float** is a fraction of the mask's equivalent-circle radius `sqrt(area / π)` (scales with object size). `0` / `0.0` = no-op |

With polygon or rectangle return formats and `mask_size_threshold < 1`,
multiple surviving blobs produce multiple regions.

Post-processing runs in order: `mask_size_threshold` → `fill_holes` →
`dilate`, so noise blobs are dropped before the mask is inflated.

### Example

```json
{
  "subpatching": { "patch_size": 1024 },
  "return_format": { "type": "PolygonLabel", "epsilon": 1, "max_points": 100 },
  "postprocess": { "mask_size_threshold": 1, "fill_holes": true, "dilate": 0 }
}
```

Ready-made files: `examples/02-extra_params/basic-rle.json` (patch + brush) and
`examples/02-extra_params/basic-polygon.json` (patch + polygon).


## Testing with `sam2plus-probe`

`sam2plus-probe` submits one image and one bounding box to a running backend
— no Label Studio server required. It re-encodes the image into the backend's
bind-mounted cache so `/predict` resolves it offline. By default it writes only
the prediction JSON to stdout.

```bash
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/sam2plus-probe [IMAGE] [--bbox X Y W H] [--extra-params FILE] [--url URL]
```

- `IMAGE` and `--bbox` default to a bundled example (COCO object 69), so
  `sam2plus-probe` with no arguments runs end to end.
- `--bbox X Y W H` — four **integers** are read as pixel `top-left + width/height`;
  four **decimals** as relative `center + width/height` (fractions of the image).
- `--extra-params FILE` — a JSON file of `extra_params` (see above), applied via
  `/setup` before `/predict`. Omit it for stock (full-image brush) behavior.
- `--url` — backend base URL (default `http://localhost:22202`).
- `--config FILE` — YAML config containing the same CLI option names.
- `--probe-fullframe-artifact-resize SCALE` — resize full-frame probe artifacts
  such as `input.jpg` and `fullframe_prediction.jpg`.

Artifact output is opt-in:

```bash
sam2plus-probe --config examples/00-probe_configs/ichthyo-rle.yaml
sam2plus-probe --config examples/00-probe_configs/ichthyo-polygon.yaml
```

With `--request-record`, `--intermediates`, `--output-img`, and `--output examples`,
artifacts are written into matching request folders:

- `examples/03-requests/<request-name>/` — input image with red prompt geometry and request JSON.
- `examples/04-intermediates/<request-name>/` — backend patch/mask intermediate graphics.
- `examples/05-outputs/<request-name>/` — returned `prediction.json`, full-frame prompt/prediction overlay, and tight prediction crops.

> `sam2plus-probe` applies `extra_params` with a `/setup` call and reads them back in
> the following `/predict` call, keyed by a shared project id. This only works
> when the backend runs a **single worker** (`WORKERS=1`, as the env files set).

Automated API tests live under `tests/`:

```bash
.venv/bin/python -m pytest tests/regular -v
.venv/bin/python -m pytest tests/bigimg -v
.venv/bin/python -m pytest tests -v
```
