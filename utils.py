import multiprocessing
import os
import traceback

import torch

device = None


def get_device():
    global device
    if device is None:
        print(f'{torch.cuda.device_count()} GPUs')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cuda')
        print(f'Using: {device}')
    return device


def get_loader(data, batch_size, shuffle=True, num_workers=8, pin=True):
    return torch.utils.data.DataLoader(dataset=data,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       pin_memory=pin,
                                       num_workers=num_workers)


def load_or_run(run_dir_path, run_name, method, *args, **kwargs):
    run_dir_path.mkdir(parents=True, exist_ok=True)
    state_path = run_dir_path / f'{run_name}@state'
    print(f'Run directory: {str(run_dir_path)}\nState file: {str(state_path)}')
    loaded = False
    if state_path.is_file():
        try:
            with state_path.open('rb') as f:
                context = torch.load(f, map_location=get_device())
                loaded = True
        except Exception:
            print(f'Exception when loading {state_path}')
            traceback.print_exc()
    if not loaded:
        context = {}
        context['model_state'] = None
        context['run_name'] = run_name
        context['run_dir'] = run_dir_path
        context['state_path'] = state_path
    context, ex = method(context, *args, **kwargs)
    if ex is not None:
        raise ex
    if 'exception' in context:
        print(context['traceback'])
    return context


def load_or_run_n(n, run_dir_path, run_name, method, *args, **kwargs):
    results = []
    for i in range(n):
        name = f'{run_name}_{i}'
        results.append(load_or_run(run_dir_path, name, method, *args, **kwargs))
    return results
