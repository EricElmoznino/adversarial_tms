import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def cv_regression(condition_features, subject):
    weights, biases, r2s, mses = [], [], [], []

    for test_conditions in subject.cv_sets:
        train_conditions = [c for c in subject.conditions if c not in test_conditions]

        train_features = torch.stack([condition_features[c] for c in train_conditions])
        test_features = torch.stack([condition_features[c] for c in test_conditions])
        train_voxels = torch.stack([subject.condition_voxels[c] for c in train_conditions])
        test_voxels = torch.stack([subject.condition_voxels[c] for c in test_conditions])

        w, b, r2, mse = regression(train_features, train_voxels, test_features, test_voxels)
        weights.append(w)
        biases.append(b)
        r2s.append(r2)
        mses.append(mse)

    mean_weight = torch.cat(weights).mean(dim=0)
    mean_bias = torch.cat(biases).mean(dim=0)
    mean_r2 = np.mean(r2s)
    mean_mse = np.mean(mses)

    return mean_weight, mean_bias, mean_r2, mean_mse


def regression(x_train, y_train, x_test, y_test):
    x_train, y_train = x_train.numpy(), y_train.numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    weight, bias = torch.from_numpy(regr.coef_), torch.from_numpy(regr.intercept_)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return weight, bias, r2, mse
