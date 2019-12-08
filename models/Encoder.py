import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, feature_extractor, regressor):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.regressor = regressor

    def forward(self, stimuli, roi_mask=None):
        features = self.feature_extractor(stimuli)
        voxels = self.regressor(features)
        if roi_mask is not None:
            roi_mask = torch.from_numpy(roi_mask.astype(np.uint8))
            voxels = voxels[:, roi_mask]
        return voxels
