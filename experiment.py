import argparse
import datetime
import math
import subprocess
import traceback
from pathlib import Path

import IPython
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from common import test_classification
from utils import get_device, load_or_run_n, loader

GIT_COMMIT = subprocess.check_output(['git', 'describe', '--always']).strip()

MODEL_SAVE_MINUTES = 5


def train_classifier(context, train_data, test_data, batch_size, batch_xs,
                     Model, model_args, init, Criterion, Optimizer,
                     optimizer_args):
    model = Model(*model_args).to(get_device())
    optimizer = Optimizer(model.parameters(), *optimizer_args)
    criterion = Criterion(reduction='mean')
    try:
        # batches_per_epoch = int(math.ceil(len(train_data) / batch_size))
        if context['model_state'] is None:
            if init is not None:
                model.apply(init)
            current_x = 0
            context['current_x'] = 0
            context['model_state'] = None
            context['optimizer_state'] = None
            context['batch_size'] = batch_size
            context['train_data_len'] = len(train_data)
            context['git_commit'] = GIT_COMMIT
        else:
            model.load_state_dict(context['model_state'])
            optimizer.load_state_dict(context['optimizer_state'])
            current_x = context['current_x']
        result_file = Path(
            context['dir_name']) / f"{context['run_name']}@state"
        result_tmp_file = Path(
            context['dir_name']) / f"{context['run_name']}@state.tmp"
        summary_writer = SummaryWriter(
            f"{context['dir_name']}/{context['run_name']}")
        # summary_writer.add_graph(model, torch.randn(()))
        train_loader = loader(train_data, batch_size)
        test_loader = loader(test_data, batch_size)
        model.train()
        model_saved = datetime.datetime.now()
        with tqdm(total=batch_xs[-1],
                  initial=current_x,
                  unit_scale=True,
                  dynamic_ncols=True) as pbar:
            while current_x <= batch_xs[-1]:
                for X, y in train_loader:
                    X = X.to(get_device())
                    y = y.to(get_device())

                    # Model evaluation
                    if current_x in batch_xs:
                        now = datetime.datetime.now()
                        pbar.set_description(
                            desc=
                            f'Last save {(now - model_saved).total_seconds():.0f}s ago',
                            refresh=False)
                        test_loss, test_acc = test_classification(model,
                                                                  test_loader,
                                                                  criterion,
                                                                  batches=10)
                        train_loss, train_acc = test_classification(
                            model, train_loader, criterion, batches=10)
                        summary_writer.add_scalar('Eval/Test loss',
                                                  test_loss,
                                                  global_step=current_x)
                        summary_writer.add_scalar('Eval/Test accuracy',
                                                  test_acc,
                                                  global_step=current_x)
                        summary_writer.add_scalar('Eval/Train loss',
                                                  train_loss,
                                                  global_step=current_x)
                        summary_writer.add_scalar('Eval/Train accuracy',
                                                  train_acc,
                                                  global_step=current_x)
                        # save model conditionally
                        if (now - model_saved
                            ).total_seconds() > 60 * MODEL_SAVE_MINUTES:
                            # save training state
                            context['current_x'] = current_x
                            context['model_state'] = model.state_dict()
                            context['optimizer_state'] = optimizer.state_dict()
                            with open(result_tmp_file, 'wb') as f:
                                torch.save(context, f)
                            result_tmp_file.replace(result_file)
                            model_saved = datetime.datetime.now()

                    # Training step
                    y_pred = model(X)
                    loss = criterion(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    summary_writer.add_scalar(f'Train/Loss',
                                              loss.item(),
                                              global_step=current_x)
                    pbar.update()
                    if current_x >= batch_xs[-1]:
                        current_x += 1
                        break
                    else:
                        current_x += 1

        if 'final_acc' not in context:
            context['current_x'] = current_x
            context['model_state'] = model.state_dict()
            context['optimizer_state'] = optimizer.state_dict()
            test_loss, test_acc = test_classification(model,
                                                      test_loader,
                                                      criterion,
                                                      batches=0)
            summary_writer.add_scalar('Eval/Test loss',
                                      test_loss,
                                      global_step=current_x)
            summary_writer.add_scalar('Eval/Test accuracy',
                                      test_acc,
                                      global_step=current_x)
            context['final_acc'] = test_acc
            context['final_loss'] = test_loss
            print(f'Final loss: {test_loss}')
            train_loss, train_acc = test_classification(model,
                                                        train_loader,
                                                        criterion,
                                                        batches=0)
            summary_writer.add_scalar('Eval/Train loss',
                                      train_loss,
                                      global_step=current_x)
            summary_writer.add_scalar('Eval/Train accuracy',
                                      train_acc,
                                      global_step=current_x)
            context['final_train_acc'] = train_acc
            context['final_train_loss'] = train_loss
            print(f'Final train loss: {train_loss}')
            # save model to secondary storage
            with open(result_tmp_file, 'wb') as f:
                torch.save(context, f)
            result_tmp_file.replace(result_file)
            result_tmp_file.unlink()
    except KeyboardInterrupt as e:
        return context, e
    except Exception as e:
        context['exception'] = e
        context['traceback'] = traceback.format_exc()
    return context, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',
                        help='Directory containing the datasets.',
                        type=Path,
                        default=Path.home() / '.datasets')
    parser.add_argument('--shell',
                        help='Spawn IPython shell after completion',
                        action='store_true')
    args = parser.parse_args()

    get_device()

    # dataset = 'mnist'
    dataset = 'cifar'

    # model = 'fc'
    model = 'conv'
    # model = 'resnet'
    # model = 'vgg'

    if model == 'fc':
        if dataset == 'mnist':
            from models import FCNet
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, ), std=(0.5, )),
            ])
            train_data = datasets.MNIST(args.dataset_dir,
                                        train=True,
                                        download=True,
                                        transform=transform)
            test_data = datasets.MNIST(args.dataset_dir,
                                       train=False,
                                       download=True,
                                       transform=transform)
            Model = FCNet
            orig_model_args = [28 * 28, 1, 3, 200, 10]
            max_epoch = 40
            batch_size = 128
        elif dataset == 'cifar':
            from models import FCNet, init_weights
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            train_data = datasets.CIFAR10(args.dataset_dir,
                                          train=True,
                                          download=True,
                                          transform=transform_train)
            test_data = datasets.CIFAR10(args.dataset_dir,
                                         train=False,
                                         download=True,
                                         transform=transform_test)
            Model = FCNet
            orig_model_args = [32 * 32, 3, 5, 200, 10]
            max_epoch = 150
            batch_size = 128
    elif model == 'conv':
        if dataset == 'mnist':
            from models import DCNet, init_weights
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, ), std=(0.5, )),
            ])
            train_data = datasets.MNIST(args.dataset_dir,
                                        train=True,
                                        download=True,
                                        transform=transform)
            test_data = datasets.MNIST(args.dataset_dir,
                                       train=False,
                                       download=True,
                                       transform=transform)
            Model = DCNet
            orig_model_args = [28, 1, 3, 30, 5, 10]
            max_epoch = 50
            batch_size = 128
        elif dataset == 'cifar':
            from models import DCNet, init_weights
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            train_data = datasets.CIFAR10(args.dataset_dir,
                                          train=True,
                                          download=True,
                                          transform=transform_train)
            test_data = datasets.CIFAR10(args.dataset_dir,
                                         train=False,
                                         download=True,
                                         transform=transform_test)
            Model = DCNet
            orig_model_args = [32, 3, 7, 40, 5, 10]
            max_epoch = 150
            batch_size = 128
    elif model == 'resnet':
        if dataset == 'cifar':
            from models import BasicBlock, ResNet, init_weights
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            train_data = datasets.CIFAR10(args.dataset_dir,
                                          train=True,
                                          download=True,
                                          transform=transform_train)
            test_data = datasets.CIFAR10(args.dataset_dir,
                                         train=False,
                                         download=True,
                                         transform=transform_test)
            Model = ResNet
            orig_model_args = [BasicBlock, [18, 18, 18], 10]
            max_epoch = 150
            batch_size = 128
        else:
            raise ValueError('TODO implement more datasets')

    Criterion = torch.nn.NLLLoss

    batches = math.ceil(len(train_data) / batch_size)
    max_batch = max_epoch * batches
    xs = [round(x) for x in np.linspace(0, max_batch - 1, num=600).tolist()]
    print(f'checkpoints: {xs}')

    Optimizer = torch.optim.Adam
    adam_betas = (0.9, 0.999)

    orig_optimizer_args = [1e-3, adam_betas]

    # ===============
    dir_name = f'n_runs_{dataset}_{model}'
    normalization = False
    init_name = init_weights.__name__ if init_weights is not None else 'None'
    # original baseline
    optimizer_args = tuple(orig_optimizer_args)
    model_args = tuple(orig_model_args)
    key = f'{Model.__name__}_args_{model_args}_init_{init_weights.__name__}_' \
          f'{Optimizer.__name__}_args_{optimizer_args}_bs_{batch_size}'
    load_or_run_n(2, dir_name, key, train_classifier, train_data, test_data,
                  batch_size, xs, Model, model_args, init_weights, Criterion,
                  Optimizer, optimizer_args)
    # ===============

    if args.shell:
        IPython.embed()


if __name__ == '__main__':
    main()
