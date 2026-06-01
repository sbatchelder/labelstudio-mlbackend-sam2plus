import time

from PIL import Image

from tests.conftest import (
    IMAGE_H,
    IMAGE_W,
    MODEL_VERSION,
    assert_fresh,
    post_predict,
    side_effect_path,
)


def test_bigimg_keypoint_prompt_returns_brush_rle(bigimg_project):
    context = {
        "result": [{
            "original_width": IMAGE_W,
            "original_height": IMAGE_H,
            "image_rotation": 0,
            "from_name": "tag3",
            "to_name": "image",
            "type": "keypointlabels",
            "is_positive": 1,
            "value": {"x": 50.0, "y": 50.0, "width": 0.5, "keypointlabels": ["probe"]},
        }],
    }

    started = time.time()
    resp = post_predict(bigimg_project, context)
    assert resp.status_code == 200, resp.text

    results = resp.json()["results"]
    assert len(results) == 1
    prediction = results[0]
    assert prediction["model_version"] == MODEL_VERSION

    region = prediction["result"][0]
    assert region["type"] == "brushlabels"
    assert region["value"]["format"] == "rle"
    assert isinstance(region["value"]["rle"], list) and region["value"]["rle"]
    assert region["value"]["brushlabels"] == ["probe"]

    assert_fresh(side_effect_path(".patch.jpg"), started)
    assert_fresh(side_effect_path(".patch.bbox.jpg"), started)
    assert_fresh(side_effect_path(".mask.patch.jpg"), started)
    assert_fresh(side_effect_path(".mask.jpg"), started)


def test_bigimg_rectangle_prompt_writes_crop_side_effects(bigimg_project):
    context = {
        "result": [{
            "original_width": IMAGE_W,
            "original_height": IMAGE_H,
            "image_rotation": 0,
            "from_name": "tag4",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": 45.0, "y": 45.0, "width": 6.0, "height": 6.0,
                "rectanglelabels": ["probe"],
            },
        }],
    }

    started = time.time()
    resp = post_predict(bigimg_project, context)
    assert resp.status_code == 200, resp.text

    results = resp.json()["results"]
    assert len(results) == 1
    region = results[0]["result"][0]
    assert region["type"] == "brushlabels"
    assert region["value"]["format"] == "rle"
    assert isinstance(region["value"]["rle"], list) and region["value"]["rle"]
    assert region["value"]["brushlabels"] == ["probe"]

    for suffix in (".bbox.jpg", ".patch.jpg", ".patch.bbox.jpg",
                   ".mask.patch.jpg", ".mask.jpg"):
        assert_fresh(side_effect_path(suffix), started)

    with Image.open(side_effect_path(".mask.patch.jpg")) as mask:
        assert mask.getbbox() is not None, "rectangle prompt produced an empty mask"


def test_bigimg_polygon_mode_returns_polygonlabels(bigimg_polygon_project):
    context = {
        "result": [{
            "original_width": IMAGE_W,
            "original_height": IMAGE_H,
            "image_rotation": 0,
            "from_name": "tag4",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": 38.15, "y": 24.52, "width": 2.3, "height": 3.4,
                "rectanglelabels": ["probe"],
            },
        }],
    }

    started = time.time()
    resp = post_predict(bigimg_polygon_project, context)
    assert resp.status_code == 200, resp.text

    results = resp.json()["results"]
    assert len(results) == 1
    prediction = results[0]
    assert prediction["model_version"] == MODEL_VERSION
    assert prediction["result"]

    region = prediction["result"][0]
    assert region["type"] == "polygonlabels"
    assert region["value"]["closed"] is True
    assert region["value"]["polygonlabels"] == ["probe"]
    assert 3 <= len(region["value"]["points"]) <= 50

    assert_fresh(side_effect_path(".patch.mask_polygon.jpg"), started)
