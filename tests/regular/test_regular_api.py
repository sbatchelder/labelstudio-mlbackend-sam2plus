from tests.conftest import IMAGE_H, IMAGE_W, MODEL_VERSION, post_predict


def test_health(backend):
    import requests

    resp = requests.get(f"{backend}/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json().get("status") == "UP"


def test_predict_without_context_returns_empty(regular_project):
    resp = post_predict(regular_project, context={})
    assert resp.status_code == 200
    assert resp.json() == {"results": []}


def test_regular_rectangle_prompt_returns_brush_rle(regular_project):
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
                "rectanglelabels": ["defect"],
            },
        }],
    }

    resp = post_predict(regular_project, context)
    assert resp.status_code == 200, resp.text

    results = resp.json()["results"]
    assert len(results) == 1
    prediction = results[0]
    assert prediction["model_version"] == MODEL_VERSION
    assert isinstance(prediction["score"], (int, float))

    region = prediction["result"][0]
    assert region["type"] == "brushlabels"
    assert region["original_width"] == IMAGE_W
    assert region["original_height"] == IMAGE_H
    assert region["value"]["format"] == "rle"
    assert isinstance(region["value"]["rle"], list) and region["value"]["rle"]
    assert region["value"]["brushlabels"] == ["defect"]
