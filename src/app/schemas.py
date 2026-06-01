from dataclasses import dataclass, field
import json
from json import JSONDecodeError
from typing import Any, Dict, Literal, Tuple, Union

Dims = Tuple[int, int]
ResizeSpec = Union[float, int, Dims]


@dataclass(frozen=True)
class SubpatchingConfig:
    patch_size: Union[int, Dims] = 1024
    mode: Literal["default", "padding"] = "default"
    padding_fill: Union[str, Tuple[int, int, int]] = "black"
    allow_oversize: bool = True
    oversize_padding: float = 0.05

    def cropper_kwargs(self) -> Dict[str, Any]:
        return {
            "crop_size": self.patch_size,
            "mode": self.mode,
            "padding_fill": self.padding_fill,
            "allow_oversize": self.allow_oversize,
            "oversize_padding": self.oversize_padding,
        }


@dataclass(frozen=True)
class ReturnFormatConfig:
    type: str = "BrushLabels"
    tag: str = "BrushLabels"
    result_type: str = "brushlabels"
    label_key: str | None = "brushlabels"
    epsilon: float | None = None
    max_points: int | None = None

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "type": self.type,
            "tag": self.tag,
            "result_type": self.result_type,
            "label_key": self.label_key,
        }
        if self.epsilon is not None:
            out["epsilon"] = self.epsilon
        if self.max_points is not None:
            out["max_points"] = self.max_points
        return out


@dataclass(frozen=True)
class PostprocessConfig:
    mask_size_threshold: float
    fill_holes: bool
    dilate: int | float = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mask_size_threshold": self.mask_size_threshold,
            "fill_holes": self.fill_holes,
            "dilate": self.dilate,
        }


@dataclass(frozen=True)
class ExtraParamsConfig:
    fullframe_resize: ResizeSpec | None = None
    subpatching: SubpatchingConfig | None = None
    return_format: ReturnFormatConfig = field(default_factory=ReturnFormatConfig)
    postprocess: PostprocessConfig | None = None


_RETURN_FORMATS = {
    "brushlabels": {
        "type": "BrushLabels",
        "tag": "BrushLabels",
        "result_type": "brushlabels",
        "label_key": "brushlabels",
    },
    "brush": {
        "type": "Brush",
        "tag": "Brush",
        "result_type": "brush",
        "label_key": None,
    },
    "polygonlabels": {
        "type": "PolygonLabels",
        "tag": "PolygonLabels",
        "result_type": "polygonlabels",
        "label_key": "polygonlabels",
    },
    "polygon": {
        "type": "Polygon",
        "tag": "Polygon",
        "result_type": "polygon",
        "label_key": None,
    },
    "rectanglelabels": {
        "type": "RectangleLabels",
        "tag": "RectangleLabels",
        "result_type": "rectanglelabels",
        "label_key": "rectanglelabels",
    },
    "rectangle": {
        "type": "Rectangle",
        "tag": "Rectangle",
        "result_type": "rectangle",
        "label_key": None,
    },
}

_POLYGON_TYPES = {"PolygonLabels", "Polygon"}
_POLYGON_DEFAULTS = {"epsilon": 1.0, "max_points": 100}

_TOP_LEVEL_KEYS = {"fullframe_resize", "subpatching", "return_format", "postprocess"}
_SUBPATCHING_KEYS = {
    "patch_size",
    "mode",
    "padding_fill",
    "allow_oversize",
    "oversize_padding",
}
_RETURN_FORMAT_KEYS = {"type", "epsilon", "max_points"}
_POSTPROCESS_KEYS = {"mask_size_threshold", "fill_holes", "dilate"}


def decode_extra_params(extra_params: Dict[str, Any] | str | None) -> Dict[str, Any]:
    if not extra_params:
        return {}
    if isinstance(extra_params, str):
        try:
            extra_params = json.loads(extra_params)
        except JSONDecodeError as exc:
            raise ValueError(
                "extra_params must be valid JSON: "
                f"{exc.msg} at line {exc.lineno} column {exc.colno}"
            ) from exc
    if not isinstance(extra_params, dict):
        raise ValueError("extra_params must be a JSON object")
    return extra_params


def normalize_extra_params(extra_params: Dict[str, Any] | str | None) -> ExtraParamsConfig:
    extra = _normalize_keys(decode_extra_params(extra_params), "extra_params")
    _reject_unknown("extra_params", extra, _TOP_LEVEL_KEYS)

    fullframe_resize = (
        _parse_fullframe_resize(extra["fullframe_resize"])
        if "fullframe_resize" in extra
        else None
    )
    subpatching = (
        _parse_subpatching(extra["subpatching"])
        if "subpatching" in extra
        else None
    )
    return_format = _parse_return_format(extra.get("return_format") or {})
    postprocess = _parse_postprocess(
        extra["postprocess"] if "postprocess" in extra else None,
        return_format.type,
    )
    return ExtraParamsConfig(
        fullframe_resize=fullframe_resize,
        subpatching=subpatching,
        return_format=return_format,
        postprocess=postprocess,
    )


def _normalize_keys(value: Any, path: str) -> Any:
    if isinstance(value, dict):
        out = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).replace("-", "_")
            if key in out:
                raise ValueError(
                    f"{path} contains duplicate keys after dash normalization: {key!r}"
                )
            out[key] = _normalize_keys(raw_value, f"{path}.{key}")
        return out
    if isinstance(value, list):
        return [_normalize_keys(item, f"{path}[]") for item in value]
    return value


def _reject_unknown(path: str, values: Dict[str, Any], allowed: set[str]):
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unsupported {path} key(s): {', '.join(unknown)}")


def _require_object(path: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _parse_subpatching(value: Any) -> SubpatchingConfig:
    cfg = _require_object("extra_params.subpatching", value)
    _reject_unknown("extra_params.subpatching", cfg, _SUBPATCHING_KEYS)
    patch_size = cfg.get("patch_size", 1024)
    _validate_patch_size(patch_size)
    if isinstance(patch_size, (list, tuple)):
        patch_size = (patch_size[0], patch_size[1])
    mode = cfg.get("mode", "default")
    if mode not in {"default", "padding"}:
        raise ValueError("extra_params.subpatching.mode must be 'default' or 'padding'")
    allow_oversize = cfg.get("allow_oversize", True)
    if not isinstance(allow_oversize, bool):
        raise ValueError("extra_params.subpatching.allow_oversize must be a boolean")
    oversize_padding = cfg.get("oversize_padding", 0.05)
    _require_number("extra_params.subpatching.oversize_padding", oversize_padding)
    return SubpatchingConfig(
        patch_size=patch_size,
        mode=mode,
        padding_fill=cfg.get("padding_fill", "black"),
        allow_oversize=allow_oversize,
        oversize_padding=oversize_padding,
    )


def _parse_return_format(value: Any) -> ReturnFormatConfig:
    cfg = _require_object("extra_params.return_format", value)
    _reject_unknown("extra_params.return_format", cfg, _RETURN_FORMAT_KEYS)
    key = str(cfg.get("type", "BrushLabels")).lower()
    if key not in _RETURN_FORMATS:
        raise ValueError(f"unsupported return_format.type: {cfg.get('type')!r}")

    base = dict(_RETURN_FORMATS[key])
    is_polygon = base["type"] in _POLYGON_TYPES
    supplied_polygon_keys = set(cfg) & {"epsilon", "max_points"}
    if supplied_polygon_keys and not is_polygon:
        keys = ", ".join(sorted(supplied_polygon_keys))
        raise ValueError(
            f"extra_params.return_format key(s) only valid for Polygon/PolygonLabels: {keys}"
        )

    if is_polygon:
        polygon_cfg = dict(_POLYGON_DEFAULTS)
        polygon_cfg.update({k: cfg[k] for k in supplied_polygon_keys})
        _require_number("extra_params.return_format.epsilon", polygon_cfg["epsilon"])
        _require_int("extra_params.return_format.max_points", polygon_cfg["max_points"])
        if polygon_cfg["epsilon"] <= 0:
            raise ValueError("extra_params.return_format.epsilon must be positive")
        if polygon_cfg["max_points"] < 3:
            raise ValueError("extra_params.return_format.max_points must be at least 3")
        base.update(polygon_cfg)

    return ReturnFormatConfig(**base)


def _parse_postprocess(value: Any, return_type: str) -> PostprocessConfig | None:
    supplied = value is not None
    cfg = _require_object("extra_params.postprocess", value) if supplied else {}
    _reject_unknown("extra_params.postprocess", cfg, _POSTPROCESS_KEYS)

    is_polygon = return_type in _POLYGON_TYPES
    values = {
        "mask_size_threshold": 0.0,
        "fill_holes": is_polygon,
        "dilate": 0,
    }
    values.update(cfg)

    if not supplied and not is_polygon:
        return None

    _require_number(
        "extra_params.postprocess.mask_size_threshold",
        values["mask_size_threshold"],
    )
    if values["mask_size_threshold"] < 0:
        raise ValueError("extra_params.postprocess.mask_size_threshold must be non-negative")
    if not isinstance(values["fill_holes"], bool):
        raise ValueError("extra_params.postprocess.fill_holes must be a boolean")
    _require_number("extra_params.postprocess.dilate", values["dilate"])

    if is_polygon and not values["fill_holes"]:
        raise ValueError(
            "postprocess.fill_holes must be true when return_format.type is "
            "Polygon/PolygonLabels (a polygon ring cannot represent a hole)"
        )
    return PostprocessConfig(**values)


def _validate_patch_size(value: Any):
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("extra_params.subpatching.patch_size must be positive")
        return
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("extra_params.subpatching.patch_size values must be positive")
        return
    raise ValueError(
        "extra_params.subpatching.patch_size must be an integer or [width, height]"
    )


def _parse_fullframe_resize(value: Any) -> ResizeSpec:
    path = "extra_params.fullframe_resize"
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a positive number or [width, height]")
    if isinstance(value, float):
        if value <= 0 or value > 1:
            raise ValueError(f"{path} float scale must be > 0 and <= 1")
        return value
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{path} must be positive")
        return value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError(f"{path} values must be positive")
        return (value[0], value[1])
    raise ValueError(f"{path} must be a float scale, integer size, or [width, height]")


def _require_number(path: str, value: Any):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")


def _require_int(path: str, value: Any):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
