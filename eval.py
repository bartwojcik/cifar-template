import torch

from utils import get_device


def test_classification(model, data_loader, criterion_class, batches=0, eval=True, device=get_device()):
    criterion = criterion_class(reduction='sum')
    if eval:
        model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct, total = 0, 0
        for batch, (X, y) in enumerate(data_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_pred = model(X)
            y_pred_max = y_pred.argmax(dim=1)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            correct += (y_pred_max == y).sum().item()
            total += y.size(0)
            if batch >= batches > 0:
                break
    if eval:
        model.train()
    # loss, acc
    return running_loss / total, correct / total