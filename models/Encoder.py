from torch import nn


class Encoder(nn.Module):

    def __init__(self, feature_extractor, regressor):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.regressor = regressor

    def forward(self, stimuli):
        features = self.feature_extractor(stimuli)
        voxels = self.regressor(features)
        return voxels
