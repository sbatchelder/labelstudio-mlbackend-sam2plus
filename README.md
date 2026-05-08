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
- customize crop area (pixel size, fraction-increase of input bbox)
- mask cleanup options like single-mask, fill gaps, grow/smooth edges
- allow multiple labelstudio instances to use endpoint
- Labelstudio Host, API Key, host-port, target GPU as .env variables.
- easy endpoint testing from within project but outside running container

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
