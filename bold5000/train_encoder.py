from argparse import ArgumentParser
import os
import math
import random
from tqdm import tqdm
import numpy as np
import torch
import utils
from models import Encoder, AlexNet, VGG16, RegressionModel
from bold5000.regression import grad_regression, lstsq_regression

random.seed(27)
resolution = 375


def voxel_data(subj_file, rois):
    voxels = np.load(subj_file, allow_pickle=True).item()
    voxels = {c: [v[r] for r in rois] for c, v in voxels.items()}
    voxels = {c: np.concatenate(v) for c, v in voxels.items()}
    voxels = {c: torch.from_numpy(v) for c, v in voxels.items()}
    return voxels


def condition_features(stimuli_folder, model):
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
            batch_feats = model(batch).cpu()
        for name, feats in zip(batch_names, batch_feats):
            condition_features[name] = feats
    return condition_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Encoder using object2vec study data')
    parser.add_argument('--bold5000_folder', required=True, type=str, help='folder containing the stimuli images')
    parser.add_argument('--feature_extractor', default='alexnet', type=str, help='feature extraction model')
    parser.add_argument('--feature_name', default='pool', type=str, help='feature extraction layer')
    parser.add_argument('--rois', nargs='+', default=['LOC', 'PPA'], type=str, help='ROIs to fit')
    args = parser.parse_args()

    if args.feature_extractor == 'alexnet':
        feat_extractor = AlexNet(args.feature_name)
    elif args.feature_extractor == 'vgg16':
        feat_extractor = VGG16(args.feature_name)
    else:
        raise ValueError('unimplemented feature extractor: {}'.format(args.feature_extractor))
    if torch.cuda.is_available():
        feat_extractor.cuda()

    voxels = voxel_data(os.path.join(args.bold5000_folder, 'subj1.npy'), args.rois)
    features = condition_features(os.path.join(args.bold5000_folder, 'stimuli'), feat_extractor)

    conditions = list(features.keys())
    random.shuffle(conditions)
    features = torch.stack([features[c] for c in conditions])
    voxels = torch.stack([voxels[c] for c in conditions])

    cv_folds = 5
    cv_features = features.split(math.ceil(features.shape[0] / cv_folds))
    cv_voxels = voxels.split(math.ceil(voxels.shape[0] / cv_folds))

    weights = []
    rs = []
    for fold in range(cv_folds):
        train_features = torch.cat([cv_features[i] for i in range(cv_folds) if i != fold])
        train_voxels = torch.cat([cv_voxels[i] for i in range(cv_folds) if i != fold])
        test_features = cv_features[fold]
        test_voxels = cv_voxels[fold]

        w, r = grad_regression(train_features, train_voxels, test_features, test_voxels)
        weights.append(w)
        rs.append(r)

    print('\nMean r score: {:.4f}'.format(np.mean(rs)))
    weight = torch.stack(weights).mean(dim=0)
    regressor = RegressionModel(features.shape[1], voxels.shape[1])
    regressor.set_params(weight)

    encoder = Encoder(feat_extractor, regressor)
    encoder.eval()
    run_name = utils.get_run_name('bold5000', args.feature_extractor, args.feature_name, args.rois)
    torch.save(encoder, os.path.join('saved_models', run_name + '.pth'))
