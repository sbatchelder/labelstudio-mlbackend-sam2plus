import sys
from pathlib import Path

import pytest

APP_SRC = Path(__file__).resolve().parents[1] / "src" / "app"
sys.path.insert(0, str(APP_SRC))

from schemas import normalize_extra_params  # noqa: E402


def test_empty_extra_params_uses_stock_like_defaults_without_optional_steps():
    cfg = normalize_extra_params({})

    assert cfg.subpatching is None
    assert cfg.return_format.type == "BrushLabels"
    assert cfg.postprocess is None


def test_extra_params_accepts_dashed_keys_and_case_insensitive_plural_labels():
    cfg = normalize_extra_params(
        """
        {
          "subpatching": {
            "patch-size": [900, 800],
            "allow-oversize": false
          },
          "return-format": {
            "type": "polygonlabels",
            "epsilon": 0.003
          },
          "postprocess": {
            "mask-size-threshold": 1,
            "fill-holes": true
          }
        }
        """
    )

    assert cfg.subpatching.patch_size == (900, 800)
    assert cfg.subpatching.allow_oversize is False
    assert cfg.return_format.type == "PolygonLabels"
    assert cfg.return_format.epsilon == 0.003
    assert cfg.return_format.max_points == 100
    assert cfg.postprocess.fill_holes is True


@pytest.mark.parametrize("return_type", ["BrushLabel", "brushlabel", "PolygonLabel"])
def test_extra_params_rejects_singular_label_return_types(return_type):
    with pytest.raises(ValueError, match="unsupported return_format.type"):
        normalize_extra_params({"return_format": {"type": return_type}})


def test_extra_params_rejects_unknown_nested_keys():
    with pytest.raises(ValueError, match="unsupported extra_params.subpatching key"):
        normalize_extra_params({"subpatching": {"patch_size": 1024, "bogus": True}})


def test_extra_params_accepts_fullframe_and_patch_size_specs():
    cfg = normalize_extra_params({
        "fullframe-resize": 0.5,
        "subpatching": {
            "patch-size": 1024,
        },
    })

    assert cfg.fullframe_resize == 0.5
    assert cfg.subpatching.patch_size == 1024


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2048, 2048),
        ([2048, 1024], (2048, 1024)),
    ],
)
def test_extra_params_accepts_fullframe_resize_pixel_specs(value, expected):
    cfg = normalize_extra_params({"fullframe_resize": value})

    assert cfg.fullframe_resize == expected


@pytest.mark.parametrize("value", [0, -1, 1.2, True])
def test_extra_params_rejects_bad_fullframe_resize_specs(value):
    with pytest.raises(ValueError, match="extra_params.fullframe_resize"):
        normalize_extra_params({"fullframe_resize": value})


def test_extra_params_rejects_removed_patch_resize():
    with pytest.raises(ValueError, match="unsupported extra_params.subpatching key"):
        normalize_extra_params({"subpatching": {"patch_resize": [512, 384]}})


def test_extra_params_rejects_malformed_json_with_location():
    with pytest.raises(ValueError, match="extra_params must be valid JSON"):
        normalize_extra_params('{"subpatching": ')


def test_extra_params_rejects_known_but_irrelevant_keys_when_supplied():
    with pytest.raises(ValueError, match="only valid for Polygon/PolygonLabels"):
        normalize_extra_params({"return_format": {"type": "BrushLabels", "epsilon": 1}})


def test_extra_params_polygon_defaults_do_not_count_as_supplied_keys():
    cfg = normalize_extra_params({"return_format": {"type": "PolygonLabels"}})

    assert cfg.return_format.epsilon == 1.0
    assert cfg.return_format.max_points == 100


def test_extra_params_polygon_implicitly_fills_holes_without_postprocess_key():
    cfg = normalize_extra_params({"return_format": {"type": "PolygonLabels"}})

    assert cfg.subpatching is None
    assert cfg.postprocess.fill_holes is True
    assert cfg.postprocess.mask_size_threshold == 0.0
    assert cfg.postprocess.dilate == 0


def test_extra_params_return_format_only_does_not_enable_subpatching_or_postprocess():
    cfg = normalize_extra_params({"return_format": {"type": "RectangleLabels"}})

    assert cfg.subpatching is None
    assert cfg.return_format.type == "RectangleLabels"
    assert cfg.postprocess is None


def test_extra_params_polygon_implicitly_fills_holes_when_postprocess_key_omits_it():
    cfg = normalize_extra_params({
        "return_format": {"type": "PolygonLabels"},
        "postprocess": {"dilate": 0.05},
    })

    assert cfg.postprocess.fill_holes is True
    assert cfg.postprocess.dilate == 0.05


def test_extra_params_polygon_rejects_explicit_fill_holes_false():
    with pytest.raises(ValueError, match="postprocess.fill_holes must be true"):
        normalize_extra_params({
            "return_format": {"type": "PolygonLabels"},
            "postprocess": {"fill-holes": False},
        })
