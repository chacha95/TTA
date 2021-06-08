from typing import List
import subprocess

from utils.util import create_random_port


class Detectron2Header(object):
    @classmethod
    def test(cls, args):
        args = args
        dist_url = create_random_port()
        command = cls.tta(args)
        subprocess.Popen(command)
        command = cls.generate_test_command(args, dist_url)
        subprocess.Popen(command)

    @classmethod
    def tta(cls, args) -> List[str]:
        script = f'{args.root_dir}/script/change_func.sh'
        command = ["/bin/bash", script]
        return command

    @classmethod
    def generate_test_command(cls, args, dist_url: str) -> List[str]:
        dt_src = f'{args.root_dir}/core/train_net.py'
        command = ["python3", dt_src,
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

        # augmentation options
        if args.TTA is True:
            command.extend(["TEST.AUG.ENABLED", "True"])

        return command
