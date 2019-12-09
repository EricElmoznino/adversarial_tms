from torch import nn


class RegressionModel(nn.Module):

    def __init__(self, n_features, n_voxels):
        super().__init__()
        self.linear = nn.Linear(n_features, n_voxels, bias=False)

    def forward(self, features):
        return self.linear(features)

    def get_params(self):
        return self.linear.weight.data

    def set_params(self, weight):
        self.linear.weight.data = weight
