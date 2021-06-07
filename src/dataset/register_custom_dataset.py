# -*- coding: utf-8 -*-

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from os.path import join


def regist_custom_dataset(dataset_dir):
    for d in ["train", "val", "test"]:
        json_file = join(dataset_dir, "annotations", d + ".json")
        image_root = join(dataset_dir, d)
        name = "custom_{}".format(d)
        # BoxMode.XYXY_ABS -> 0
        # BoxMode.XYWH_ABS -> 1
        register_coco_instances(name, {"bbox_mode": 1}, json_file, image_root)
