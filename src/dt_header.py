import subprocess
from utils.util import create_random_port
from tta import TTA


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
        TTA.set_flip(args.aug)
        TTA.set_multi_scale(args.multi_scale)
        TTA.set_color_trans(args.color_trans)
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
                   "TEST.AUG.ENABLED", args.aug,
                   "DATALOADER.NUM_WORKERS", args.num_worker,
                   "MODEL.ROI_HEADS.NUM_CLASSES", args.num_class,
                   "INPUT.CROP.SIZE", "[1.0, 1.0]",
                   "CUDNN_BENCHMARK", "False"]
        return command
