"""Shared HTTP integration-test helpers for SAM2Plus."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_BACKEND_URL = os.environ.get("ML_BACKEND_URL", "http://localhost:22202").rstrip("/")
CACHE_DIR = Path(os.environ.get("CACHE_DIR", REPO_ROOT / "data" / "cache"))
BOOT_TIMEOUT = float(os.environ.get("ML_BACKEND_BOOT_TIMEOUT", "300"))

EXAMPLE_IMAGE = REPO_ROOT / "examples" / "01-data" / "ichthyoliths.jpg"
IMAGE_W, IMAGE_H = 10246, 9818
IMAGE_URL = "http://sam2plus.test/ichthyoliths.jpg"

MODEL_VERSION = "0.0.2"
REGULAR_PROJECT = "tests-regular"
BIGIMG_PROJECT = "tests-bigimg"
BIGIMG_POLYGON_PROJECT = "tests-bigimg-polygon"

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="probe"/>
  </BrushLabels>
  <PolygonLabels name="tag2" toName="image">
    <Label value="probe"/>
  </PolygonLabels>
  <KeyPointLabels name="tag3" toName="image" smart="true">
    <Label value="probe"/>
  </KeyPointLabels>
  <RectangleLabels name="tag4" toName="image" smart="true">
    <Label value="probe"/>
  </RectangleLabels>
</View>
"""


def cache_stem() -> str:
    digest = hashlib.md5(IMAGE_URL.encode(), usedforsecurity=False).hexdigest()[:8]
    filename = os.path.basename(urlparse(IMAGE_URL).path)
    return f"{digest}__{filename}"[: -len(".jpg")]


def side_effect_path(suffix: str) -> Path:
    return CACHE_DIR / f"{cache_stem()}{suffix}"


def post_setup(project: str, extra_params: dict):
    return requests.post(
        f"{ML_BACKEND_URL}/setup",
        json={
            "project": project,
            "schema": LABEL_CONFIG,
            "extra_params": json.dumps(extra_params),
        },
        timeout=BOOT_TIMEOUT,
    )


def post_predict(project: str, context: dict, task_id=1):
    body = {
        "project": project,
        "tasks": [{"id": task_id, "data": {"image": IMAGE_URL}}],
        "label_config": LABEL_CONFIG,
        "params": {"context": context},
    }
    return requests.post(f"{ML_BACKEND_URL}/predict", json=body, timeout=BOOT_TIMEOUT)


def assert_fresh(path: Path, since: float):
    assert path.exists(), f"expected the backend to write {path}"
    assert path.stat().st_mtime >= since - 1, f"{path} was not refreshed by this request"
    with Image.open(path) as im:
        im.verify()


@pytest.fixture(scope="session")
def backend():
    deadline = time.time() + BOOT_TIMEOUT
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{ML_BACKEND_URL}/health", timeout=5)
            if resp.status_code == 200:
                return ML_BACKEND_URL
        except requests.RequestException as exc:
            last_err = exc
        time.sleep(3)
    pytest.skip(
        f"ML backend not healthy at {ML_BACKEND_URL} within {BOOT_TIMEOUT}s "
        f"(last error: {last_err}). Start it with compose first."
    )


@pytest.fixture(scope="session")
def seeded_image(backend):
    assert EXAMPLE_IMAGE.exists(), f"missing example image: {EXAMPLE_IMAGE}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{cache_stem()}.jpg"
    shutil.copyfile(EXAMPLE_IMAGE, dest)
    return dest


@pytest.fixture()
def regular_project(backend, seeded_image):
    resp = post_setup(REGULAR_PROJECT, {})
    assert resp.status_code == 200, resp.text
    return REGULAR_PROJECT


@pytest.fixture()
def bigimg_project(backend, seeded_image):
    resp = post_setup(BIGIMG_PROJECT, {"subpatching": {"patch_size": 1024}})
    assert resp.status_code == 200, resp.text
    return BIGIMG_PROJECT


@pytest.fixture()
def bigimg_polygon_project(backend, seeded_image):
    resp = post_setup(
        BIGIMG_POLYGON_PROJECT,
        {
            "subpatching": {"patch_size": 1024},
            "postprocess": {
                "mask_size_threshold": 1,
                "fill_holes": True,
                "dilate": 0.05,
            },
            "return_format": {
                "type": "PolygonLabel",
                "epsilon": 0.003,
                "max_points": 50,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    return BIGIMG_POLYGON_PROJECT
