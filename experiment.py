import argparse
from pathlib import Path

import torch

from datasets import get_cifar10, get_mnist
from models import DCNet, FCNet
from train import train
from utils import load_or_run_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
                        help='directory to save the results to',
                        type=Path,
                        default=Path.cwd() / 'runs')
    parser.add_argument('--dataset_dir',
                        help='directory containing the datasets',
                        type=Path,
                        default=Path.home() / '.datasets')
    args = parser.parse_args()

    # run_name = f'mnist_simple'
    # settings = {}
    # settings['model_class'] = FCNet
    # settings['model_args'] = {'image_size': 28, 'channels': 1, 'num_layers': 3, 'layer_size': 200, 'classes': 10}
    # settings['param_init_fun'] = None
    # settings['train_data'], settings['train_eval_data'], settings['test_data'] = get_mnist(args.dataset_dir)
    # settings['batch_size'] = 128
    # settings['optimizer_class'] = torch.optim.Adam
    # settings['optimizer_args'] = {'lr': 1e-3, 'betas': (0.9, 0.999)}
    # settings['criterion_type'] = torch.nn.CrossEntropyLoss
    # settings['epochs'] = 40
    # settings['eval_points'] = 400

    run_name = f'cifar_simple'
    settings = {}
    settings['model_class'] = DCNet
    settings['model_args'] = {
        'image_size': 32,
        'channels': 3,
        'num_layers': 7,
        'num_filters': 50,
        'kernel_size': 5,
        'classes': 10,
        'batchnorm': True
    }
    settings['param_init_fun'] = None
    settings['train_data'], settings['train_eval_data'], settings['test_data'] = get_cifar10(args.dataset_dir)
    settings['batch_size'] = 128
    settings['optimizer_class'] = torch.optim.Adam
    settings['optimizer_args'] = {'lr': 1e-4, 'betas': (0.9, 0.999)}
    settings['criterion_type'] = torch.nn.CrossEntropyLoss
    settings['epochs'] = 150
    settings['eval_points'] = 400

    load_or_run_n(2, args.results_dir, run_name, train, settings)


if __name__ == '__main__':
    main()
