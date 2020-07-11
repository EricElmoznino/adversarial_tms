import random
import torch
from torch import nn
from torch import optim
from models import RegressionModel

random.seed(27)


def lstsq_regression(x_train, y_train, x_test, y_test):
    u, s, v = torch.svd(x_train)
    s_inv = (1. / s).view(1, s.size(0))
    vs = v * s_inv  # inverse of diagonal is just reciprocal of diagonal
    uty = torch.mm(u.permute(1, 0), y_train)
    w = torch.mm(vs, uty)

    y_pred = torch.mm(x_test, w)
    r = correlation(y_pred, y_test)

    return w, r


def grad_regression(x_train, y_train, x_test, y_test):
    model = RegressionModel(x_train.shape[1], y_train.shape[1])
    if torch.cuda.is_available():
        model.cuda()
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    batch_size = 32
    n_epochs = 50
    indices = list(range(x_train.shape[0]))
    for epoch in range(n_epochs):
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i: i + batch_size]
            x_batch, y_batch = x_train[batch_indices], y_train[batch_indices]
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_mse = []
        test_r = []
        for i in range(0, x_test.shape[0], batch_size):
            x_batch, y_batch = x_test[i:i + batch_size], y_test[i:i + batch_size]
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            with torch.no_grad():
                y_pred = model(x_batch)

            mse = loss_func(y_pred, y_batch)
            r = correlation(y_pred, y_batch)
            test_mse.append(mse)
            test_r.append(r)

        mean_mse = torch.stack(test_mse).mean().item()
        mean_r = torch.stack(test_r).mean().item()
        print('[{} / {}]\tMSE: {:.3f}\tr: {:.3f}'.format(epoch, n_epochs, mean_mse, mean_r))

    print('\nFinished regression')

    return model.get_params(), mean_r


def correlation(a, b, mean=True):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean(axis=0)
    if mean:
        r = r.mean()
    return r
