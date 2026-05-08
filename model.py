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


class SAM2_BigImg(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        logging.info('SAM2_BigImg setup')
        self.set("model_version", "0.0.1")

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


    def _sam_predict(self, img_url, point_coords=None, point_labels=None, input_box=None, task=None):
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
        
        logger.debug(f'extra_params: {self.extra_params} ({type(self.extra_params)})')
            
        cropper = CropMapper(image, **self.extra_params)
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
        #np.save(mask_path_cache_cropped, mask_cropped)
        Image.fromarray(mask_cropped * 255).save(mask_path_cache_cropped.replace('.npy','.jpg'))
        mask = cropper.mask_crop_to_full(mask_cropped)
        #np.save(mask_path_cache, mask)
        Image.fromarray(mask * 255).save(mask_path_cache.replace('.npy','.jpg'))
        return {
            'masks': [mask],
            'probs': [prob]
        }


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask for a smart keypoint that has been placed."""
        logger.info(f'MODEL predict(tasks={tasks}, context={context})')
        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

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
            task=tasks[0]
        )

        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label)
        
        return ModelResponse(predictions=predictions)
