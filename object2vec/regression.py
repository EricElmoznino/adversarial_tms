import numpy as np
import torch
from sklearn.linear_model import LinearRegression


def cv_regression(condition_features, subject):
    weights, biases, rmats = [], [], []

    for test_conditions in subject.cv_sets:
        train_conditions = [c for c in subject.conditions if c not in test_conditions]

        train_features = np.stack([condition_features[c].numpy() for c in train_conditions])
        test_features = np.stack([condition_features[c].numpy() for c in test_conditions])
        train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
        test_voxels = np.stack([subject.condition_voxels[c] for c in test_conditions])

        w, b, rmat = regression(train_features, train_voxels, test_features, test_voxels)
        weights.append(w)
        biases.append(b)
        rmats.append(rmat)

    mean_weight = np.stack(weights).mean(axis=0)
    mean_bias = np.stack(biases).mean(axis=0)
    mean_weight = torch.from_numpy(mean_weight)
    mean_bias = torch.from_numpy(mean_bias)
    mean_r = np.mean(rmats)

    return mean_weight, mean_bias, mean_r


def regression(x_train, y_train, x_test, y_test):

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    weight, bias = regr.coef_, regr.intercept_

    zs = lambda v: (v - v.mean(0)) / v.std(0)
    rmat = (zs(y_test) * zs(y_pred)).mean(axis=0)

    return weight, bias, rmat
