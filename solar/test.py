from torch.utils.data import DataLoader
import torch
from LSTM import *

def test(model, partition, args):
    testloader = DataLoader(partition['test'], 
                           batch_size=args.batch_size, 
                           shuffle=False, drop_last=True)
    model.eval()

    test_acc = 0.0
    with torch.no_grad():
        for i, (X) in enumerate(testloader):

            X = X[:, :, 8:9].transpose(0, 1).float().to(args.device)
            # y_true = y[:, :, 8].float().to(args.device)
            model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            # test_acc += metric(y_pred, y_true)[0]

    # test_acc = test_acc / len(testloader)
    return y_pred #test_acc