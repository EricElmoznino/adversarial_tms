from torch import nn


class RegressionModel(nn.Module):

    def __init__(self, n_features, n_voxels):
        super().__init__()
        self.linear = nn.Linear(n_features, n_voxels)

    def forward(self, features):
        return self.linear(features)
