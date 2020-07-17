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


class PCAEncoder(nn.Module):

    def __init__(self, feature_extractor, pcs, mean):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.projection = nn.Linear(pcs.shape[1], pcs.shape[0], bias=False)
        self.projection.weight.data = torch.from_numpy(pcs)
        self.mean = nn.Parameter(data=torch.from_numpy(mean))

    def forward(self, stimuli):
        features = self.feature_extractor(stimuli)
        centered_features = features - self.mean
        pcs = self.projection(centered_features)
        return pcs


class CCAEncoder(nn.Module):

    def __init__(self, pca_encoder, rotations):
        super().__init__()

        self.pca_encoder = pca_encoder
        self.rotation = nn.Linear(rotations.shape[1], rotations.shape[0], bias=False)
        self.rotation.weight.data = torch.from_numpy(rotations.T)

    def forward(self, stimuli):
        pcs = self.pca_encoder(stimuli)
        transformed = self.rotation(pcs)
        return transformed
