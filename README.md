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

# SAM2BigImg

This is a modification of the [Labelstudio SAM2 ml-backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image). It can be used to interactively add mask (brush) annotations on Labelstudio. 

This fork allows for the creation of masks for small objects within a very large input image.
The when an image is received by the ml-endpoint, an area around the Labelstudio annotator's point/bounding-box prompt is cropped from the original image. The SAM2 mask is generated using the crop, and the resulting mask's coordinates are then remapped to the original image's coordinates and returned to Labelstudio. 
Without this, masks for small objects in large scenes loose definition, since SAM2 resizes input images to WxH resolution. With this method, mask resolution is preserved. 

## TODOs
- ~~customize crop area (pixel size, fraction-increase of input bbox)~~ — done: `crop_size` / `oversize_padding` (see [Model parameters](#model-parameters-extra_params))
- mask cleanup: ~~single-mask, fill gaps~~ done via `postprocess`; grow/smooth edges still open
- allow multiple labelstudio instances to use endpoint
- Labelstudio Host, API Key, host-port, target GPU as .env variables.
- ~~easy endpoint testing from within project but outside running container~~ — done: `probe.py`

## Configuration

In `compose.yml`, update container name to avoid collisions with other ML-Backends and provide your LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY (legacy tokens only).

## Installation
```
docker compose build
docker compose up -d
```

## Initial Project Setup Notes

```
mkdir SAM2BigImg && cd SAM2BigImg
git init
git remote add upstream https://github.com/HumanSignal/label-studio-ml-backend.git
git fetch upstream master
git archive upstream/master label_studio_ml/examples/segment_anything_2_image | tar -x --strip-components=3
git add -A && git commit -m "init sam2 ml-backend"
```
Then update `model.py`, incl. renaming for `class NewModel` to `class SAM2_BigImg` and subsequent changes in `_wsgi.py`. 


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


## Running with Docker

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.


## Configuration
Parameters can be set in `docker-compose.yml` before running the container.


The following common parameters are available:
- `DEVICE` - specify the device for the model server (currently only `cuda` is supported, `cpu` is coming soon)
- `MODEL_CONFIG` - SAM2 model configuration file (`sam2_hiera_l.yaml` by default)
- `MODEL_CHECKPOINT` - SAM2 model checkpoint file (`sam2_hiera_large.pt` by default)
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server


## Model parameters (`extra_params`)

With **no** `extra_params` the backend behaves exactly like the stock
[SAM2 image backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image):
the full image is sent to SAM2 and a brush (RLE) mask is returned. Supplying
**any** `extra_params` switches on the BigImg pipeline — crop around the prompt,
optionally post-process, and optionally return polygons instead of brush masks.

`extra_params` are static per-project values. Label Studio sends them to the
backend's `/setup` endpoint; for local testing `probe.py` does the same via
`--extra-args` (see below). Unknown keys are logged as warnings and ignored.

### Crop parameters

Forwarded to `CropMapper` (`seg_cropper.py`):

| Key | Default | Meaning |
|-----|---------|---------|
| `crop_size` | `1024` | crop edge length in px — an int (square) or `[width, height]` |
| `mode` | `"default"` | `default` keeps the crop inside the image bounds; `padding` centers it on the prompt and pads past edges |
| `padding_fill` | `"black"` | fill color used by `padding` mode |
| `allow_size_override` | `true` | grow the crop when the prompt box is larger than `crop_size` |
| `oversize_padding` | `0.05` | extra fractional margin added when the crop is grown to fit an oversized box |

### Polygon output (`as_polygon`)

Set `as_polygon` to return Label Studio **`polygonlabels`** regions instead of
brush masks. The value is `true` (use defaults) or an object:

| Key | Default | Meaning |
|-----|---------|---------|
| `epsilon` | `1` | Douglas–Peucker simplification tolerance. `< 1` is treated as a fraction of the contour perimeter (resolution-independent); `>= 1` is an absolute distance in crop pixels. Larger = fewer, coarser vertices. |
| `max_points` | `100` | hard cap on vertices per polygon; `epsilon` is raised by a binary search until the cap is met |

Polygon output requires a `<PolygonLabels>` control tag in the labeling config
(in addition to, or instead of, `<BrushLabels>`). It also shifts the
`postprocess` defaults — see below.

### Post-processing (`postprocess`)

Applied to the binary mask before it is turned into a brush mask or polygons:

| Key | Default (brush / polygon) | Meaning |
|-----|---------------------------|---------|
| `mask_size_threshold` | `0` / `1` | keep connected components whose area ≥ threshold × the largest component. `1` = largest blob only; `0.5` = blobs at least half its size; `0` = keep everything |
| `fill_holes` | `false` / `true` | fill the interior holes of each blob. Must be `true` when `as_polygon` is set (a polygon ring cannot encode a hole) — otherwise the request errors |

With `as_polygon` and `mask_size_threshold < 1`, multiple surviving blobs
produce multiple polygon regions.

### Example

```json
{
  "crop_size": 1024,
  "as_polygon": { "epsilon": 1, "max_points": 100 },
  "postprocess": { "mask_size_threshold": 1, "fill_holes": true }
}
```

Ready-made files: `examples/basic-rle.json` (crop + brush) and
`examples/basic-polygon.json` (crop + polygon).


## Testing with `probe.py`

`probe.py` submits one image and one bounding box to a running backend and
saves the prediction JSON — no Label Studio server required. It re-encodes the
image into the backend's bind-mounted cache so `/predict` resolves it offline.

```bash
pip install -r requirements-test.txt
python probe.py [IMAGE] [--bbox X Y W H] [-o OUTDIR] [--extra-args FILE] [--url URL]
```

- `IMAGE` and `--bbox` default to a bundled example (COCO object 69), so
  `python probe.py` with no arguments runs end to end.
- `--bbox X Y W H` — four **integers** are read as pixel `top-left + width/height`;
  four **decimals** as relative `center + width/height` (fractions of the image).
- `--extra-args FILE` — a JSON file of `extra_params` (see above), applied via
  `/setup` before `/predict`. Omit it for stock (full-image brush) behavior.
- `--url` — backend base URL (default `http://localhost:22202`).

Outputs land in `OUTDIR` (`probe_out/` by default): `prediction_rle.json` or
`prediction_polygon.json` (by mode), plus copies of the cache side-effect
images — `bbox.jpg`, `bbox.crop.jpg`, `crop.mask_rle.jpg`, and, in polygon
mode, `crop.mask_polygon.jpg` (the crop with the polygon drawn in red).

> `probe.py` applies `extra_params` with a `/setup` call and reads them back in
> the following `/predict` call, keyed by a shared project id. This only works
> when the backend runs a **single worker** (`WORKERS=1`, as docker-compose sets).

Automated API tests live in `test_api.py` (`pytest test_api.py`).
