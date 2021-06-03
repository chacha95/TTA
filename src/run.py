import argparse
import os
import sys
root_dir = os.getcwd()
sys.path.insert(1, root_dir)

from dt_header import Detectron2Header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTA')
    parser.add_argument('--root_dir', type=str, default=root_dir, help='root directory')
    parser.add_argument('--data_path', type=str, default='/mnt/dataset-storage/COCO-2017', help='dataset path')
    parser.add_argument('--output_dir', type=str, default=f'{root_dir}/model', help='output path')
    parser.add_argument('--model_weights', type=str, default=f'{root_dir}/model/model-R101-FPN-3x.pkl', help='model weight path')
    parser.add_argument('--model_config', type=str, default=f'{root_dir}/model/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
                        help='model config file path')
    parser.add_argument('--test_dataset', type=str, default="('coco_2017_val',)", help='custom test dataset')
    parser.add_argument('--eval_only', type=str, default="True", help='test mode(evaluation mode)')
    parser.add_argument('--num_worker', type=str, default='4', help='number of cpu threads')
    parser.add_argument('--num_class', type=str, default='80', help='number of classes in given dataset')
    parser.add_argument('--num_gpu', type=str, default='1', help='number of gpu for training')
    parser.add_argument('--batch_size', type=str, default='1', help='number of batch')
    args = parser.parse_args()

    Detectron2Header.test(args)
