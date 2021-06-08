import argparse


def get_args():
    parser = argparse.ArgumentParser(description='TTA')
    parser.add_argument('--root_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_weights')
    parser.add_argument('--model_config')
    parser.add_argument('--test_dataset')
    parser.add_argument('--num_worker')
    parser.add_argument('--num_class')
    parser.add_argument('--num_gpu')
    parser.add_argument('--batch_size')
    parser.add_argument('--TTA')
    parser.add_argument('--multi_scale')
    parser.add_argument('--flip')
    parser.add_argument('--contrast')

    return parser.parse_args()
