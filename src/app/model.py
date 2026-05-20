import torch
import numpy as np
import os
import sys
import json
import pathlib
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from PIL import Image, ImageDraw

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from seg_cropper import CropMapper, Box
from postprocess import (filter_components, fill_holes, mask_to_polygons,
                         draw_polygons, POLYGON_DEFAULTS)

import logging
logger=logging.getLogger(__name__)

DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam2_checkpoint = str(os.path.join(ROOT_DIR, "checkpoints", MODEL_CHECKPOINT))

sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)

predictor = SAM2ImagePredictor(sam2_model)

logger.info('MODEL BUILD_SAM2')


def box_xwyh2xyxy(box:Box) -> Box:
    x,w,y,h = box
    return x,y,x+w,y+h


# extra_params keys forwarded to CropMapper; everything else is handled here.
_CROPMAPPER_KEYS = {'crop_size', 'mode', 'padding_fill', 'allow_size_override',
                    'oversize_padding'}


def parse_extra_params(extra_params):
    """Split extra_params into (crop_kwargs, polygon_cfg | None, postprocess_cfg).

    - crop_kwargs    : kwargs forwarded to CropMapper
    - polygon_cfg    : {'epsilon', 'max_points'} when as_polygon is set, else None
    - postprocess_cfg: {'mask_size_threshold', 'fill_holes'}

    Unknown top-level keys are warned about and ignored. With as_polygon the
    postprocess defaults shift (size threshold 1, fill_holes true) and
    fill_holes=false becomes an error (a polygon ring cannot encode a hole).
    """
    extra = dict(extra_params or {})
    as_polygon = extra.pop('as_polygon', None)
    postprocess = extra.pop('postprocess', None) or {}

    crop_kwargs = {}
    for key, value in extra.items():
        if key in _CROPMAPPER_KEYS:
            crop_kwargs[key] = value
        else:
            logger.warning(f'ignoring unknown extra_params key: {key!r}')

    polygon_cfg = None
    if as_polygon:
        polygon_cfg = dict(POLYGON_DEFAULTS)
        if isinstance(as_polygon, dict):
            polygon_cfg.update(as_polygon)

    # hyphenated keys (e.g. "fill-holes") are accepted and normalised.
    postprocess = {k.replace('-', '_'): v for k, v in dict(postprocess).items()}
    postprocess_cfg = {
        'mask_size_threshold': 1.0 if polygon_cfg else 0.0,
        'fill_holes': bool(polygon_cfg),
    }
    for key in postprocess_cfg:
        if key in postprocess:
            postprocess_cfg[key] = postprocess[key]

    if polygon_cfg and not postprocess_cfg['fill_holes']:
        raise ValueError('postprocess.fill_holes must be true when as_polygon '
                          'is set (a polygon ring cannot represent a hole)')
    return crop_kwargs, polygon_cfg, postprocess_cfg


class SAM2_BigImg(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        logging.info('SAM2_BigImg setup')
        self.set("model_version", "0.0.2")

    def get_results(self, masks, probs, width, height, from_name, to_name, label):
        logger.info('MODEL get_results')
        results = []
        total_prob = 0
        for mask, prob in zip(masks, probs):
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:4]
            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)
            total_prob += prob
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]


    def get_polygon_results(self, polygons, probs, width, height,
                            from_name, to_name, label):
        logger.info('MODEL get_polygon_results')
        results = []
        total_prob = 0
        for polygon, prob in zip(polygons, probs):
            label_id = str(uuid4())[:4]
            # Label Studio stores polygon points as percentages of the image.
            points = [[x / width * 100.0, y / height * 100.0] for x, y in polygon]
            total_prob += prob
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'points': points,
                    'closed': True,
                    'polygonlabels': [label],
                },
                'score': prob,
                'type': 'polygonlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]


    def _sam_predict(self, img_url, point_coords=None, point_labels=None, input_box=None,
                     task=None, crop_kwargs=None, polygon_cfg=None, postprocess_cfg=None):
        logger.info('MODEL _sam_predict')

        cache_dir = '/cache'
        if not os.path.isdir(cache_dir): os.mkdir(cache_dir)
        image_path = get_local_path(img_url, task_id=task.get('id'), cache_dir=cache_dir)

        image_path_cache = os.path.join('/cache',os.path.basename(image_path))
        image_path_cache_cropped = image_path_cache.replace('.jpg','.crop.jpg')
        mask_path_cache = image_path_cache.replace('.jpg','.mask.npy')
        mask_path_cache_cropped = image_path_cache.replace('.jpg','.mask.crop.npy')

        image = Image.open(image_path).convert("RGB")

        image2 = image.copy()
        draw2 = ImageDraw.Draw(image2)
        if input_box:
            draw2.rectangle(input_box, outline='red', width=5)
            image2.save(image_path_cache.replace('.jpg','.bbox.jpg'))

        logger.debug(f'crop_kwargs={crop_kwargs} polygon_cfg={polygon_cfg} '
                     f'postprocess_cfg={postprocess_cfg}')

        cropper = CropMapper(image, **(crop_kwargs or {}))
        if input_box:
            img = cropper.crop(box=input_box)
            img2 = img.copy()
            input_box = (input_box[0] - cropper.offx,
                         input_box[1] - cropper.offy,
                         input_box[2] - cropper.offx,
                         input_box[3] - cropper.offy)
            logger.debug(f'offset_box: {input_box}')
            draw1 = ImageDraw.Draw(img2)
            draw1.rectangle(input_box, outline='red', width=5)
            img2.save(image_path_cache_cropped.replace('.crop.','.crop.bbox.'))
        else:
            logger.info(point_coords)
            img = cropper.crop(point_coords)  # todo check these dont neeed to be inverse
        img.save(image_path_cache_cropped)

        img = np.array(img)
        predictor.set_image(img)

        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        # logits = logits[sorted_ind]
        sorted_ind = np.argsort(scores)[::-1]
        scores = scores[sorted_ind]
        prob = float(scores[0])
        masks = masks[sorted_ind]
        mask_cropped = masks[0, :, :].astype(np.uint8)

        # Post-process the crop-resolution mask. Under brush defaults
        # (threshold 0, fill_holes false) both steps are no-ops.
        postprocess_cfg = postprocess_cfg or {'mask_size_threshold': 0.0,
                                              'fill_holes': False}
        mask_cropped = filter_components(mask_cropped,
                                         postprocess_cfg['mask_size_threshold'])
        if postprocess_cfg['fill_holes']:
            mask_cropped = fill_holes(mask_cropped)
        Image.fromarray(mask_cropped * 255).save(mask_path_cache_cropped.replace('.npy','.jpg'))

        if polygon_cfg:
            # Polygon path: work entirely from the crop mask -- no full-image
            # mask, no RLE. `img` is the crop's RGB array (set above).
            polygons = mask_to_polygons(mask_cropped,
                                        epsilon=polygon_cfg['epsilon'],
                                        max_points=polygon_cfg['max_points'])
            overlay = draw_polygons(img, polygons)
            Image.fromarray(overlay).save(
                image_path_cache.replace('.jpg', '.crop.mask_polygon.jpg'))
            polygons_full = cropper.polygons_crop_to_full(polygons)
            logger.info(f'MODEL _sam_predict: {len(polygons_full)} polygon(s)')
            return {
                'polygons': polygons_full,
                'probs': [prob] * len(polygons_full)
            }

        mask = cropper.mask_crop_to_full(mask_cropped)
        Image.fromarray(mask * 255).save(mask_path_cache.replace('.npy','.jpg'))
        return {
            'masks': [mask],
            'probs': [prob]
        }


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask/polygon for a smart prompt that was placed."""
        logger.info(f'MODEL predict(tasks={tasks}, context={context})')
        crop_kwargs, polygon_cfg, postprocess_cfg = parse_extra_params(self.extra_params)
        control_tag = 'PolygonLabels' if polygon_cfg else 'BrushLabels'
        from_name, to_name, value = self.get_first_tag_occurence(control_tag, 'Image')

        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            return ModelResponse(predictions=[])

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # collect context information
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        print(f'Point coords are {point_coords}, point labels are {point_labels}, input box is {input_box}')

        img_url = tasks[0]['data'][value]
        predictor_results = self._sam_predict(
            img_url=img_url,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0],
            crop_kwargs=crop_kwargs,
            polygon_cfg=polygon_cfg,
            postprocess_cfg=postprocess_cfg
        )

        if polygon_cfg:
            predictions = self.get_polygon_results(
                polygons=predictor_results['polygons'],
                probs=predictor_results['probs'],
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label=selected_label)
        else:
            predictions = self.get_results(
                masks=predictor_results['masks'],
                probs=predictor_results['probs'],
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label=selected_label)

        return ModelResponse(predictions=predictions)
