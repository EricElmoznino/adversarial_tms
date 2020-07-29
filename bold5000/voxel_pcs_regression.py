from argparse import ArgumentParser
import os
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import torch
import utils
from models import Encoder, AlexNet, VGG16, RegressionModel
from bold5000.regression import grad_regression


def voxel_data(subj_file, roi):
    voxels = np.load(subj_file, allow_pickle=True).item()
    voxels = {s: v[roi] for s, v in voxels.items()}
    stimuli = list(voxels.keys())
    voxels = np.stack([voxels[s] for s in stimuli])
    return voxels, stimuli


def condition_features(stimuli, model):
    print('Extracting features')
    condition_features = []
    batch_size = 32
    for i in tqdm(range(0, len(stimuli), batch_size)):
        batch_names = stimuli[i:i + batch_size]
        batch = [utils.image_to_tensor(s, resolution=256) for s in batch_names]
        batch = torch.stack(batch)
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            batch_feats = model(batch).cpu().numpy()
        condition_features.append(batch_feats)
    condition_features = np.concatenate(condition_features)
    return condition_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Encoder using BOLD5000 study data')
    parser.add_argument('--bold5000_folder', required=True, type=str, help='folder containing the stimuli images')
    parser.add_argument('--roi', required=True, type=str, help='ROI to fit')
    parser.add_argument('--n_pcs', default=100, type=int, help='number of pcs to reduce voxel dimensions')
    parser.add_argument('--feature_extractor', default='alexnet', type=str, help='feature extraction model')
    parser.add_argument('--feature_name', default='conv_3', type=str, help='feature extraction layer')
    parser.add_argument('--all_subj', action='store_true', help='whether or not to use all subjects')
    parser.add_argument('--l2', default=0, type=float, help='L2 regularization weight')
    args = parser.parse_args()

    if args.feature_extractor == 'alexnet':
        feat_extractor = AlexNet(args.feature_name)
    elif args.feature_extractor == 'vgg16':
        feat_extractor = VGG16(args.feature_name)
    else:
        raise ValueError('unimplemented feature extractor: {}'.format(args.feature_extractor))
    if torch.cuda.is_available():
        feat_extractor.cuda()

    subj_file = 'subjall.npy' if args.all_subj else 'subj1.npy'
    voxels, stimuli = voxel_data(os.path.join(args.bold5000_folder, subj_file), args.roi)
    voxel_pcs = PCA(n_components=voxels.shape[1]).fit_transform(voxels)

    stimuli = [os.path.join(args.bold5000_folder, 'stimuli', s) for s in stimuli]
    features = condition_features(stimuli, feat_extractor)

    cv_r = []
    cv = KFold(n_splits=5, shuffle=True, random_state=27)
    for train_idx, val_idx in cv.split(features):
        features_train, features_val = features[train_idx], features[val_idx]
        voxel_pcs_train, voxel_pcs_val = voxel_pcs[train_idx], voxels[val_idx]
        _, r = grad_regression(features_train, voxel_pcs_train,
                               features_val, voxel_pcs_val, l2_penalty=args.l2)
        cv_r.append(r)
    print('\nFinal Mean r: {:.4f}'.format(np.mean(cv_r)))

    w, _ = grad_regression(features, voxel_pcs, l2_penalty=args.l2)
    regressor = RegressionModel(features.shape[1], voxel_pcs.shape[1])
    regressor.set_params(w)

    encoder = Encoder(feat_extractor, regressor)
    encoder.eval()
    run_name = utils.get_run_name('bold5000', args.feature_extractor, args.feature_name, [args.roi + 'pcs'], args.subj)
    torch.save(encoder, os.path.join('saved_models', run_name + '.pth'))
