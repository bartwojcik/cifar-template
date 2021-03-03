import math
import traceback
from datetime import datetime

import neptune
import torch

from eval import test_classification
from utils import get_device, get_loader

MODEL_SAVE_MINUTES = 15
BATCHES_TO_EVAL = 5


def train(context, settings):
    model = settings['model_class'](**settings['model_args']).to(get_device())
    init_fun = settings['param_init_fun']
    train_data = settings['train_data']
    train_eval_data = settings['train_eval_data']
    test_data = settings['test_data']
    batch_size = settings['batch_size']
    epochs = settings['epochs']
    optimizer = settings['optimizer_class'](model.parameters(), **settings['optimizer_args'])
    criterion_type = settings['criterion_type']
    criterion = criterion_type(reduction='mean')
    batches_per_epoch = math.ceil(len(train_data) / batch_size)
    last_batch = settings['epochs'] * batches_per_epoch - 1
    eval_batches = [
        round(x) for x in torch.linspace(0, last_batch, steps=settings['eval_points'], device='cpu').tolist()
    ]
    train_loader = get_loader(train_data, batch_size)
    train_eval_loader = get_loader(train_eval_data, batch_size)
    test_loader = get_loader(test_data, batch_size)
    try:
        if context['model_state'] is None:
            if init_fun is not None:
                model.apply(init_fun)
            current_batch = 0
            context['current_batch'] = 0
            # context['model_state'] = None
            context['optimizer_state'] = None
        else:
            model.load_state_dict(context['model_state'])
            optimizer.load_state_dict(context['optimizer_state'])
            current_batch = context['current_x']
        state_path = context['state_path']
        tmp_state_path = state_path.parent / f'{state_path.name}.tmp'
        neptune.create_experiment(name=context['run_name'], params=settings, upload_source_files='**/*.py')
        model_saved = datetime.now()
        model.train()
        train_iter = iter(train_loader)
        assert last_batch in eval_batches
        while current_batch <= last_batch:
            try:
                X, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, y = next(train_iter)

            X = X.to(get_device(), non_blocking=True)
            y = y.to(get_device(), non_blocking=True)

            # Model evaluation
            if current_batch in eval_batches:
                neptune.log_metric('Progress', current_batch, current_batch / last_batch)
                now = datetime.now()
                test_loss, test_acc = test_classification(model, test_loader, criterion_type, batches=BATCHES_TO_EVAL)
                neptune.log_metric('Eval/Test loss', current_batch, test_loss)
                neptune.log_metric('Eval/Test accuracy', current_batch, test_acc)
                train_loss, train_acc = test_classification(model,
                                                            train_eval_loader,
                                                            criterion_type,
                                                            batches=BATCHES_TO_EVAL)
                neptune.log_metric('Eval/Train loss', current_batch, train_loss)
                neptune.log_metric('Eval/Train accuracy', current_batch, train_acc)
                # save model conditionally
                if (now - model_saved).total_seconds() > 60 * MODEL_SAVE_MINUTES:
                    context['current_x'] = current_batch
                    context['model_state'] = model.state_dict()
                    context['optimizer_state'] = optimizer.state_dict()
                    with open(tmp_state_path, 'wb') as f:
                        torch.save(context, f)
                    tmp_state_path.replace(state_path)
                model_saved = datetime.now()

            # Training step
            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            neptune.log_metric('Train/Loss', current_batch, loss.item())
            current_batch += 1

        if 'completed' not in context:
            context['current_x'] = current_batch
            context['model_state'] = model.state_dict()
            context['optimizer_state'] = optimizer.state_dict()
            test_loss, test_acc = test_classification(model, test_loader, criterion_type)
            neptune.log_metric('Eval/Test loss', current_batch, test_loss)
            neptune.log_metric('Eval/Test accuracy', current_batch, test_acc)
            context['final_acc'] = test_acc
            context['final_loss'] = test_loss
            print(f'Final loss: {test_loss}\nFinal acc: {test_acc}')
            train_loss, train_acc = test_classification(model, train_eval_loader, criterion_type)
            neptune.log_metric('Eval/Train loss', current_batch, train_loss)
            neptune.log_metric('Eval/Train accuracy', current_batch, train_acc)
            context['final_train_acc'] = train_acc
            context['final_train_loss'] = train_loss
            print(f'Final train loss: {train_loss}\nFinal train acc: {train_acc}')
            context['completed'] = True
            # save model to secondary storage
            with open(tmp_state_path, 'wb') as f:
                torch.save(context, f)
            tmp_state_path.replace(state_path)
            # save model to neptune
            neptune.log_artifact(str(tmp_state_path))
    except KeyboardInterrupt as e:
        return context, e
    except Exception as e:
        context['exception'] = e
        context['traceback'] = traceback.format_exc()
    return context, None
