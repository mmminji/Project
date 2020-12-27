from torch.utils.data import DataLoader
from LSTM import *

def train(model, partition, optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'], 
                             batch_size=args.batch_size, 
                             shuffle=True, drop_last=True)

    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    train_acc = 0.0
    train_loss = 0.0
    for i, (X, y) in enumerate(trainloader):

        # X : [n, 10, 6] input_len, batch_size, input_dim
        # Y : [10, m] batch_size, output_len
        X = X[:, :, 8:9].transpose(0, 1).float().to(args.device)   # np.swapaxes와 같은 역할, batch_size와 seq_len의 위치를 바꿔줌(위에가 원하는 형태)
        y_true = y[:, :, 8].float().to(args.device)   # close만 예측하려고 3번째 선택
        #print(torch.max(X[:, :, 3]), torch.max(y_true))

        model.zero_grad()
        optimizer.zero_grad()
        model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

        y_pred = model(X)
        loss = loss_fn(y_pred.view(-1), y_true.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += metric(y_pred, y_true)[0]

    train_loss = train_loss / len(trainloader)
    train_acc = train_acc / len(trainloader)
    return model, train_loss, train_acc