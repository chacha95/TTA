import argparse
import sys
root_dir = "/home/appuser/src"
sys.path.insert(1, root_dir)
from dt_header import Detectron2Header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTA')
    parser.add_argument('--root_dir', type=str, default=root_dir, help='root directory')
    parser.add_argument('--output_dir', type=str, default=f'{root_dir}/model', help='output path')
    parser.add_argument('--model_weights', type=str, default=f'{root_dir}/model/faster_rcnn_R_50_FPN_3x.pkl',
                        help='model weight path')
    parser.add_argument('--model_config', type=str, default=f'{root_dir}/model/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='model config file path')
    parser.add_argument('--test_dataset', type=str, default="('coco2017val',)", help='custom test dataset')
    parser.add_argument('--num_worker', type=str, default='4', help='number of cpu threads')
    parser.add_argument('--num_class', type=str, default='80', help='number of classes in given dataset')
    parser.add_argument('--num_gpu', type=str, default='1', help='number of gpu for training')
    parser.add_argument('--batch_size', type=str, default='1', help='number of batch')
    # ----------------------------------------TTA options----------------------------------------
    parser.add_argument('--TTA', type=bool, default=True, help='Augmentation option')
    parser.add_argument('--flip', type=bool, default=True, help='horizontal flip option')
    parser.add_argument('--contrast', type=bool, default=True, help='contrast transformation')
    parser.add_argument('--multi_scale', type=bool, default=True, help='multi scale sizes')
    # -------------------------------------------------------------------------------------------
    args = parser.parse_args()

    Detectron2Header.test(args)
