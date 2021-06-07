import subprocess
import argparse
import os
import sys
root_dir = os.getcwd()
sys.path.insert(1, root_dir)
from utils.util import create_random_port


class Detectron2Header:
    @classmethod
    def test(cls, args):
        args = args
        # open port
        dist_url = create_random_port()
        # TTA
        command = cls.tta(args)
        subprocess.Popen(command)

        # test command
        command = cls.generate_test_command(args, dist_url)
        subprocess.Popen(command)

    @classmethod
    def tta(cls, args):
        script = f'{args.root_dir}/script/change_func.sh'
        command = ["/bin/bash", script]
        return command

    @classmethod
    def generate_test_command(cls, args, dist_url):
        dt_src = f'{args.root_dir}/core/train_net.py'
        command = ["python3", dt_src,
                   "--datapath", args.datapath,
                   "--num-gpus", args.num_gpu,
                   "--config-file", args.model_config,
                   "--dist-url", dist_url,
                   "--eval-only", "MODEL.WEIGHTS", args.model_weights,
                   "OUTPUT_DIR", args.output_dir,
                   "DATASETS.TEST", args.test_dataset,
                   "DATALOADER.NUM_WORKERS", args.num_worker,
                   "MODEL.ROI_HEADS.NUM_CLASSES", args.num_class,
                   "INPUT.CROP.SIZE", "[1.0, 1.0]",
                   "CUDNN_BENCHMARK", "False"]
        return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTA')
    parser.add_argument('--root_dir', type=str, default=root_dir, help='root directory')
    parser.add_argument('--datapath', type=str, default='/home/appuser/dataset/COCO2017/', help='dataset path')
    parser.add_argument('--output_dir', type=str, default=f'{root_dir}/model', help='output path')
    parser.add_argument('--model_weights', type=str, default=f'{root_dir}/model/faster_rcnn_R_50_FPN_3x.pkl',
                        help='model weight path')
    parser.add_argument('--model_config', type=str, default=f'{root_dir}/model/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='model config file path')
    parser.add_argument('--test_dataset', type=str, default="('coco2017val',)", help='custom test dataset')
    parser.add_argument('--eval_only', type=str, default="True", help='test mode(evaluation mode)')
    parser.add_argument('--num_worker', type=str, default='4', help='number of cpu threads')
    parser.add_argument('--num_class', type=str, default='80', help='number of classes in given dataset')
    parser.add_argument('--num_gpu', type=str, default='1', help='number of gpu for training')
    parser.add_argument('--batch_size', type=str, default='1', help='number of batch')
    args = parser.parse_args()

    Detectron2Header.test(args)
