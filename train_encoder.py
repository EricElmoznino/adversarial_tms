import os
import torch
from helpers import utils
from models import AlexNet, RegressionModel


def mean_condition_features(stimuli_folder, model):
    conditions = utils.listdir(stimuli_folder)
    features = []
    for c in conditions:
        stimuli = utils.listdir(c)
        stimuli = [utils.image_to_tensor(s) for s in stimuli]
        stimuli = torch.stack(stimuli)
        feats = model(stimuli).mean(dim=0)
        features.append(feats)
    return features
