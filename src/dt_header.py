import subprocess
from utils.util import create_random_port


class Detectron2Header:
    @classmethod
    def test(cls, args):
        args = args
        dist_url = create_random_port()
        command = cls.generate_test_command(args, dist_url)
        p = subprocess.Popen(command)
        p.communicate()

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
                   "TEST.AUG.ENABLED", args.aug,
                   "MODEL.ROI_HEADS.NUM_CLASSES", args.num_class,
                   "INPUT.CROP.SIZE", "[1.0, 1.0]",
                   "CUDNN_BENCHMARK", "False"]
        return command
