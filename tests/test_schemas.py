import sys
from pathlib import Path

import pytest

APP_SRC = Path(__file__).resolve().parents[1] / "src" / "app"
sys.path.insert(0, str(APP_SRC))

from schemas import normalize_extra_params  # noqa: E402


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

    assert cfg.subpatching.patch_size == [900, 800]
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
