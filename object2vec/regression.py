import numpy as np
import torch
from sklearn.linear_model import LinearRegression


def cv_regression(condition_features, subject):
    weights, rmats = [], []

    for test_conditions in subject.cv_sets:
        train_conditions = [c for c in subject.conditions if c not in test_conditions]

        train_features = np.stack([condition_features[c].numpy() for c in train_conditions])
        test_features = np.stack([condition_features[c].numpy() for c in test_conditions])
        train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
        test_voxels = np.stack([subject.condition_voxels[c] for c in test_conditions])

        w, rmat = regression(train_features, train_voxels, test_features, test_voxels)
        weights.append(w)
        rmats.append(rmat)

    mean_weight = np.stack(weights).mean(axis=0)
    mean_weight = torch.from_numpy(mean_weight)
    mean_r = np.mean(rmats)

    return mean_weight, mean_r


def regression(x_train, y_train, x_test, y_test):
    regr = LinearRegression(fit_intercept=False)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    weight = regr.coef_
    rmat = correlation(y_test, y_pred)
    return weight, rmat


def correlation(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean(0)
    return r
