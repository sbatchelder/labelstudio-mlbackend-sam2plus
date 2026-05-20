from typing import Iterable, List, Tuple, Union, Literal
from PIL import Image, ImageDraw
import numpy as np

import logging
logger=logging.getLogger(__name__)

Point = Tuple[int, int] # Coordinates in (x, y) format
Dims = Tuple[int, int]  # Dimensions in (width, height) format
Box = Tuple[int, int, int, int]  # Bounding box in (x0, y0, x1, y1) format

class CropMapper:
    def __init__(self, img:Image.Image, crop_size: Union[int, Dims]=1024,
                 mode: Literal['default', 'padding'] = 'default',
                 padding_fill: Union[str, Tuple[int, int, int]] = 'black',
                 allow_size_override:bool = True,
                 oversize_padding: float = 0.05):

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        if crop_size[0] <= 0 or crop_size[1] <= 0:
            raise ValueError("size must be positive")

        self.img = img
        self.size = crop_size
        self.fill = padding_fill
        self.offset: Point = (0, 0)  # Offset from full image to cropped coords
        self.mode = mode
        self.allow_size_override = allow_size_override
        self.oversize_padding = 1+oversize_padding


    @property
    def offx(self) -> int:
        return self.offset[0]
    @property
    def offy(self) -> int:
        return self.offset[1]


    def crop(self, point_coords:List[Point]=None, box:Box=None) -> Image.Image:
        if box:
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            width = box[2] - box[0]
            height = box[3] - box[1]
        elif point_coords:
            leftmost, rightmost = self.img.width, 0
            topmost, bottommost = self.img.height, 0
            for x, y in point_coords:
                leftmost = min(leftmost, x)
                rightmost = max(rightmost, x)
                topmost = min(topmost, y)
                bottommost = max(bottommost, y)
            # find center and width/height of list of Points
            center = ((leftmost + rightmost) // 2, (topmost + bottommost) // 2)
            width = rightmost - leftmost
            height = bottommost - topmost
        else:
            raise ValueError("Either point_coords or box must be provided")

        # adjust size if needed
        if width > self.size[0] or height > self.size[1]:
            if not self.allow_size_override:
                raise NotImplementedError
            largest = max(width, height)
            largest_plus = largest * self.oversize_padding  # eg 1.05 is 5% larger
            largest_plus = largest_plus + largest_plus % 2  # ensure even number
            self.size = (largest_plus, largest_plus)

        if self.mode == 'padding':
            return self.crop_with_padding(center)
        else:
            return self.crop_inside(center)


    def crop_with_padding(self, center: Point) -> Image.Image:
        """
        Returns (crop_img, offset). Offset is (x0, y0) which can be negative if padding occurs.
        The requested center is exactly at (size//2, size//2) in the cropped image.
        """
        cx, cy = center
        cw, ch = self.size
        x0 = cx - cw//2
        y0 = cy - ch//2
        x1 = x0 + cw
        y1 = y0 + ch
        self.offset = (int(x0), int(y0))

        W, H = self.img.size
        if x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H:
            return self.img.crop((x0, y0, x1, y1))

        # Pad into a canvas so crop is always s×s
        logging.debug(f'crop_with_padding mode: {self.img.mode}; cw,ch={(cw, ch)}')
        canvas = Image.new(self.img.mode, (int(cw), int(ch)), color=self.fill)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(W, x1)
        src_y1 = min(H, y1)

        patch = self.img.crop((src_x0, src_y0, src_x1, src_y1))
        dst_x = src_x0 - x0
        dst_y = src_y0 - y0
        canvas.paste(patch, (int(dst_x), int(dst_y)))
        return canvas

    def crop_inside(self, center: Point) -> Image.Image:
        """
        Returns (crop_img, offset). Offset is clamped so the s×s crop lies fully inside the image.
        If the image is smaller than crop-size in either dimension, falls back to padding variant.
        """
        W, H = self.img.size
        cw,ch = self.size
        if W < cw or H < ch:
            # Can't fit inside — use padding behavior
            return self.crop_with_padding(center)

        cx, cy = center
        x0 = max(0, min(cx - cw//2, W - cw))
        y0 = max(0, min(cy - ch//2, H - ch))
        self.offset = (int(x0), int(y0))
        return self.img.crop((x0, y0, x0 + cw, y0 + ch))


    def polygons_crop_to_full(self, polygons):
        """Map crop-coordinate polygons to full-image pixel coordinates.

        Counterpart of mask_crop_to_full() for vector geometry: shifts each
        point by the crop offset and clamps it to the image bounds (a crop in
        padding mode can extend past the edges).
        """
        W, H = self.img.size
        x0, y0 = self.offset
        mapped = []
        for polygon in polygons:
            mapped.append([
                (min(max(x + x0, 0), W), min(max(y + y0, 0), H))
                for x, y in polygon
            ])
        return mapped


    def mask_crop_to_full(self, mask_crop):
        W, H = self.img.size
        h, w = mask_crop.shape
        
        x0, y0 = self.offset

        # Initialize full mask
        mask_full = np.zeros((H, W), dtype=np.uint8)

        # Overlap between crop box and full image
        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        src_x1 = min(w, W - x0)
        src_y1 = min(h, H - y0)

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        if src_x1 > src_x0 and src_y1 > src_y0:
            mask_full[dst_y0:dst_y1, dst_x0:dst_x1] = mask_crop[src_y0:src_y1, src_x0:src_x1]

        return mask_full


