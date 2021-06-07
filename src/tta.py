# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
from contextlib import contextmanager
from itertools import count
from typing import List
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.config import configurable
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations
)
from detectron2.structures import Boxes, Instances

from .meta_arch import GeneralizedRCNN
from .postprocessing import detector_postprocess
from .roi_heads.fast_rcnn import fast_rcnn_inference_single_image

__all__ = ["DatasetMapperTTA", "GeneralizedRCNNWithTTA"]

# Define a sequence of augmentations:
# from detectron2.data import transforms as T
# augs = T.AugmentationList([
#     T.RandomBrightness(0.9, 1.1),
# ])  # type: T.Augmentation
#
# input = T.AugInput(image, boxes=boxes)
# transform = augs(input)
# image_transformed = input.image  # new image


# this parameter change!
class TTA(object):
    _flip = False
    _multi_scale_mins = []
    _multi_scale_max = None
    _color_trans = []

    # enable augmentation
    # _flip = True
    # _multi_scale_mins = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # _multi_scale_max = 4000
    # _color_trans = True

    @classmethod
    def get_multi_scale(cls):
        return cls._multi_scale_mins, cls._multi_scale_max

    @classmethod
    def get_flip(cls):
        return cls._flip

    @classmethod
    def get_color_trans(cls):
        return cls._color_trans


class DatasetMapperTTA:
    @configurable
    def __init__(self, min_sizes: List[int], max_size: int, flip: bool, color_trans: List[float]):
        """
        Args:
            min_sizes: list of short-edge size to resize the image to
            max_size: maximum height or width of resized images
            flip: whether to apply flipping augmentation
            color_trans: color transformation info
        """
        self.min_sizes = min_sizes
        self.max_size = max_size
        self.flip = flip
        self.color_trans = color_trans

    @classmethod
    def from_config(cls):
        return {
            "min_sizes": None,
            "max_size": None,
            "flip": None,
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                A list of dicts, which contain augmented version of the input image.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret


class GeneralizedRCNNWithTTA(nn.Module):
    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        self.cfg = cfg.clone()
        self.model = model
        self.batch_size = batch_size
        min_sizes, max_size = TTA.get_multi_scale()
        flip = TTA.get_flip()
        color_trans = TTA.get_color_trans()
        self.tta_mapper = DatasetMapperTTA(min_sizes, max_size, flip, color_trans)

    def __call__(self, batched_inputs):
        def _read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        return [self._inference_one_image(_read_image(x)) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        all_boxes, all_scores, all_classes = self._get_augmented_boxes(augmented_inputs, tfms)
        # merge all detected boxes to obtain final predictions for boxes
        merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)
        return {"instances": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_boxes(self, augmented_inputs, tfms):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        for output, tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            pred_boxes = output.pred_boxes.tensor
            original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
            all_boxes.append(torch.from_numpy(original_pred_boxes).to(pred_boxes.device))

            all_scores.extend(output.scores)
            all_classes.extend(output.pred_classes)
        all_boxes = torch.cat(all_boxes, dim=0)
        return all_boxes, all_scores, all_classes

    def _batch_inference(self, batched_inputs, detected_instances=None):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(
                    self.model.inference(
                        inputs,
                        instances if instances[0] is not None else None,
                        do_postprocess=False,
                    )
                )
                inputs, instances = [], []
        return outputs

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        # select from the union of all results
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            1e-8,
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        return merged_instances
