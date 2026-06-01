import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

APP_SRC = Path(__file__).resolve().parents[1] / "src" / "app"
sys.path.insert(0, str(APP_SRC))

from seg_cropper import CropMapper  # noqa: E402


def make_image(width=2000, height=1500):
    return Image.new("RGB", (width, height))


# --- crop geometry -----------------------------------------------------------

def test_crop_inside_default_mode_clamps_offset_to_bounds():
    cm = CropMapper(make_image(), crop_size=1024)
    patch = cm.crop(box=(900, 650, 1100, 850))  # 200x200, center (1000, 750)

    assert patch.size == (1024, 1024)
    # center-512 clamped into [0, W-1024] / [0, H-1024]
    assert cm.offset == (488, 238)


def test_crop_point_coords_uses_prompt_bounding_box():
    cm = CropMapper(make_image(), crop_size=1024)
    patch = cm.crop([(900, 650), (1100, 850)])  # same span as the box above

    assert patch.size == (1024, 1024)
    assert cm.offset == (488, 238)


def test_crop_requires_a_prompt():
    cm = CropMapper(make_image())
    with pytest.raises(ValueError, match="Either point_coords or box"):
        cm.crop()


def test_padding_mode_allows_negative_offset_and_keeps_patch_size():
    cm = CropMapper(make_image(), crop_size=1024, mode="padding")
    patch = cm.crop(box=(0, 0, 100, 100))  # center (50, 50)

    assert patch.size == (1024, 1024)  # padded canvas, not clamped into the image
    assert cm.offset == (50 - 512, 50 - 512)


# --- oversize growth ---------------------------------------------------------

def test_oversize_box_grows_patch_to_even_square():
    cm = CropMapper(make_image(), crop_size=1024, oversize_padding=0.05)
    patch = cm.crop(box=(400, 200, 1600, 1400))  # 1200x1200 > 1024

    # 1200 * 1.05 = 1260, already even, fits inside the 2000x1500 image
    assert cm.size == (1260, 1260)
    assert patch.size == (1260, 1260)


def test_oversize_growth_rounds_down_to_even():
    cm = CropMapper(make_image(4000, 4000), crop_size=1024, oversize_padding=0.05)
    cm.crop(box=(0, 0, 1238, 1238))  # 1238 * 1.05 = 1299.9 -> int 1299 -> 1298

    assert cm.size == (1298, 1298)


def test_allow_oversize_false_rejects_oversized_prompt():
    cm = CropMapper(make_image(), crop_size=1024, allow_oversize=False)
    with pytest.raises(ValueError, match="exceeds patch size"):
        cm.crop(box=(400, 200, 1600, 1400))


# --- exceeds_image fallback predicate ---------------------------------------

def test_exceeds_image_false_when_region_fits_patch():
    cm = CropMapper(make_image(), crop_size=1024)
    assert cm.exceeds_image(box=(900, 650, 1100, 850)) is False


def test_exceeds_image_false_for_oversize_that_still_fits_image():
    cm = CropMapper(make_image(), crop_size=1024, oversize_padding=0.05)
    # 1200 * 1.05 = 1260 <= min(2000, 1500)
    assert cm.exceeds_image(box=(400, 200, 1600, 1400)) is False


def test_exceeds_image_true_when_padded_patch_exceeds_image():
    cm = CropMapper(make_image(), crop_size=1024, oversize_padding=0.05)
    # 1450 * 1.05 = 1522.5 > min(2000, 1500)
    assert cm.exceeds_image(box=(100, 100, 1550, 300)) is True


def test_exceeds_image_supports_point_prompts():
    cm = CropMapper(make_image(), crop_size=1024, oversize_padding=0.05)
    assert cm.exceeds_image(point_coords=[(10, 10), (1990, 290)]) is True


# --- coordinate mapping back to the full image ------------------------------

def test_mask_crop_to_full_places_mask_at_offset():
    cm = CropMapper(make_image(), crop_size=1024)
    cm.crop(box=(900, 650, 1100, 850))  # offset (488, 238)

    mask_crop = np.zeros((1024, 1024), dtype=np.uint8)
    mask_crop[0:10, 0:10] = 1
    mask_full = cm.mask_crop_to_full(mask_crop)

    assert mask_full.shape == (1500, 2000)  # (H, W)
    assert mask_full.sum() == 100
    assert mask_full[238:248, 488:498].sum() == 100


def test_polygons_crop_to_full_shifts_and_clamps():
    cm = CropMapper(make_image(), crop_size=1024)
    cm.crop(box=(900, 650, 1100, 850))  # offset (488, 238)

    mapped = cm.polygons_crop_to_full([[(0, 0), (10, 0), (10, 10)]])
    assert mapped == [[(488, 238), (498, 238), (498, 248)]]


def test_boxes_crop_to_full_shifts_and_clamps():
    cm = CropMapper(make_image(), crop_size=1024)
    cm.crop(box=(900, 650, 1100, 850))  # offset (488, 238)

    mapped = cm.boxes_crop_to_full([(0, 0, 10, 10)])
    assert mapped == [(488, 238, 498, 248)]
