import multiprocessing
import os
import subprocess
import traceback
from itertools import product

import torch

# TODO investigate why >0 causes error with set_default_tensor_type
# https://discuss.pytorch.org/t/is-there-anything-wrong-with-setting-default-tensor-type-to-cuda/27949
# see also:
# https://github.com/pytorch/pytorch/issues/19996
# >0 causes high cpu usage - why?
LOADER_WORKERS = 4
PIN_MEMORY = True

device = None


def get_device():
    global device
    if device is None:
        print(f'{multiprocessing.cpu_count()} CPUs')

        print(f'{torch.cuda.device_count()} GPUs')
        if torch.cuda.is_available():
            device = 'cuda:0'
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.backends.cudnn.benchmark = True
        else:
            # torch.set_default_tensor_type(torch.FloatTensor)
            device = 'cpu'
        print(f'Using: {device}')
    return device


def loader(data, batch_size):
    return torch.utils.data.DataLoader(dataset=data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=PIN_MEMORY,
                                       num_workers=LOADER_WORKERS)


def load_or_run(dir_name, run_name, method, *args, **kwargs):
    os.makedirs(dir_name, exist_ok=True)
    filepath = os.path.join(dir_name, f'{run_name}@state')
    print(f'State file: {filepath}')
    loaded = False
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as f:
                context = torch.load(f, map_location=get_device())
                loaded = True
        except Exception:
            print(f'Exception when loading {filepath}')
            traceback.print_exc()
    if not loaded:
        context = {}
        context['model_state'] = None
        context['run_name'] = run_name
        context['dir_name'] = dir_name
    # TODO maybe move arguments into context?
    context, ex = method(context, *args, **kwargs)
    if ex is not None:
        raise ex
    if 'exception' in context:
        print(context['traceback'])
    return context


def load_or_run_n(n, dir_name, run_name, method, *args, **kwargs):
    results = []
    for i in range(n):
        name = f'{run_name}_{i}'
        results.append(load_or_run(dir_name, name, method, *args, **kwargs))
    return results
