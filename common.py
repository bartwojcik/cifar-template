import torch

from utils import get_device


def test_classification(model,
                        data_loader,
                        criterion,
                        batches=0,
                        device=get_device(),
                        eval=True):
    if eval:
        model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct, total = 0, 0
        for batch, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            y_pred_max = y_pred.argmax(dim=1)
            correct += (y_pred_max == y).sum().item()
            total += y.size(0)
            if batch >= batches > 0:
                break
    if eval:
        model.train()
    # loss, acc
    return running_loss / (batch + 1), correct / total