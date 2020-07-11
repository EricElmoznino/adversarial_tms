from argparse import ArgumentParser
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import utils
from models import RegressionModel

random.seed(27)
resolution = 256


def correlation(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean(axis=0)
    return r


def voxel_data(subj_file, rois):
    voxels = np.load(subj_file, allow_pickle=True).item()
    voxels = {c: [v[r] for r in rois] for c, v in voxels.items()}
    voxels = {c: np.concatenate(v) for c, v in voxels.items()}
    return voxels


def stimulus_predictions(stimuli_folder, model):
    print('Extracting features')
    stimuli = os.listdir(stimuli_folder)
    condition_features = {}
    batch_size = 32
    for i in tqdm(range(0, len(stimuli), batch_size)):
        batch_names = stimuli[i:i + batch_size]
        batch = [utils.image_to_tensor(os.path.join(stimuli_folder, n), resolution=resolution)
                 for n in batch_names]
        batch = torch.stack(batch)
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            batch_feats = model(batch).cpu().numpy()
        for name, feats in zip(batch_names, batch_feats):
            condition_features[name] = feats
    return condition_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Select voxels from two encoders such that their performance is matched')
    parser.add_argument('--bold5000_folder', required=True, type=str, help='folder containing the stimuli images')
    parser.add_argument('--weak_roi', required=True, type=str, help='ROI with poor encoder performance')
    parser.add_argument('--strong_roi', required=True, type=str, help='ROI with good encoder performance')
    parser.add_argument('--weak_encoder', required=True, type=str, help='path to ROI encoder with poor performance')
    parser.add_argument('--strong_encoder', required=True, type=str, help='path to ROI encoder with good performance')
    args = parser.parse_args()

    weak_encoder = torch.load(args.weak_encoder, map_location=lambda storage, loc: storage)
    strong_encoder = torch.load(args.strong_encoder, map_location=lambda storage, loc: storage)
    if torch.cuda.is_available():
        weak_encoder.cuda()
        strong_encoder.cuda()

    weak_voxels = voxel_data(os.path.join(args.bold5000_folder, 'subj1.npy'), [args.weak_roi])
    strong_voxels = voxel_data(os.path.join(args.bold5000_folder, 'subj1.npy'), [args.strong_roi])
    weak_predictions = stimulus_predictions(os.path.join(args.bold5000_folder, 'stimuli'), weak_encoder)
    strong_predictions = stimulus_predictions(os.path.join(args.bold5000_folder, 'stimuli'), strong_encoder)

    weak_predictions = np.stack([weak_predictions[c] for c in weak_voxels])
    strong_predictions = np.stack([strong_predictions[c] for c in strong_voxels])
    weak_voxels = np.stack([weak_voxels[c] for c in weak_voxels])
    strong_voxels = np.stack([strong_voxels[c] for c in strong_voxels])

    weak_r = correlation(weak_predictions, weak_voxels)
    strong_r = correlation(strong_predictions, strong_voxels)

    weak_r_order = weak_r.argsort()[::-1]
    strong_r_order = strong_r.argsort()[::-1]

    weak_i = 0
    strong_i = 0
    weak_selected = []
    strong_selected = []
    while strong_i < len(strong_r_order) and weak_i < len(weak_r_order):
        weak_idx = weak_r_order[weak_i]
        strong_idx = strong_r_order[strong_i]
        weak_idx_r = weak_r[weak_idx]
        strong_idx_r = strong_r[strong_idx]
        if strong_idx_r <= weak_idx_r:
            weak_selected.append(weak_idx)
            strong_selected.append(strong_idx)
            weak_i += 1
            strong_i += 1
        else:
            strong_i += 1

    weak_encoder.cpu()
    strong_encoder.cpu()
    weak_regressor = RegressionModel(weak_encoder.regressor.linear.in_features, len(weak_selected))
    strong_regressor = RegressionModel(strong_encoder.regressor.linear.in_features, len(strong_selected))
    weak_regressor.set_params(weak_encoder.regressor.get_params()[weak_selected, :])
    strong_regressor.set_params(strong_encoder.regressor.get_params()[strong_selected, :])
    weak_encoder.regressor = weak_regressor
    strong_encoder.regressor = strong_regressor
    torch.save(weak_encoder, args.weak_encoder.replace('.pth', '_matched.pth'))
    torch.save(strong_encoder, args.strong_encoder.replace('.pth', '_matched.pth'))
