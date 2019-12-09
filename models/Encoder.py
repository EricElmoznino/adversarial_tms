import copy
import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, feature_extractor, regressor):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.regressor = regressor
        self.roi_mask = None

    def forward(self, stimuli):
        features = self.feature_extractor(stimuli)
        voxels = self.regressor(features)
        if self.roi_mask is not None:
            voxels = voxels[:, self.roi_mask]
        return voxels

    def random_weights(self):
        rand = copy.deepcopy(self)
        weights = rand.regressor.get_params()
        rand_weights = torch.rand_like(weights)
        rand_weights = (rand_weights / rand_weights.norm()) * weights.norm()
        rand.regressor.set_params(rand_weights)
        return rand

    def set_roi_mask(self, roi_mask):
        self.roi_mask = torch.from_numpy(roi_mask.astype(np.uint8))
