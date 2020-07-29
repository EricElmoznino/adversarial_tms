import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
from models import CCAEncoder, PCAEncoder, AlexNet
import utils

n_components = 10
roi = 'PPA'
bold5000_folder = '/home/eelmozn1/datasets/adversarial_tms/bold5000'
feat_extractor = AlexNet('conv_3')


def voxel_data(subj_file, roi):
    voxels = np.load(subj_file, allow_pickle=True).item()
    voxels = {c: v[roi] for c, v in voxels.items()}
    return voxels


def condition_features(stimuli_folder, model):
    print('Extracting features')
    stimuli = os.listdir(stimuli_folder)
    condition_features = {}
    batch_size = 32
    for i in tqdm(range(0, len(stimuli), batch_size)):
        batch_names = stimuli[i:i + batch_size]
        batch = [utils.image_to_tensor(os.path.join(stimuli_folder, n), resolution=256)
                 for n in batch_names]
        batch = torch.stack(batch)
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            batch_feats = model(batch).cpu().numpy()
        for name, feats in zip(batch_names, batch_feats):
            condition_features[name] = feats
    return condition_features


def correlation(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean(axis=0)
    return r


if torch.cuda.is_available():
    feat_extractor.cuda()

voxels = voxel_data(os.path.join(bold5000_folder, 'subj1.npy'), roi)
features = condition_features(os.path.join(bold5000_folder, 'stimuli'), feat_extractor)
features = np.stack([features[c] for c in voxels], axis=0)
voxels = np.stack([voxels[c] for c in voxels], axis=0)

pca = PCA(n_components=400)
pca.fit(features)
pcs = pca.transform(features)
print('\nPCA Mean Explained Variance: {:.4f}'.format(np.mean(pca.explained_variance_ratio_.mean())))
pca_encoder = PCAEncoder(feat_extractor, pcs=pca.components_, mean=pca.mean_)

cca = CCA(n_components=n_components, scale=False)

cv = KFold(n_splits=5, shuffle=True, random_state=27)
cv_train_r, cv_val_r = [], []
for train_idx, val_idx in cv.split(pcs):
    pcs_train, pcs_val = pcs[train_idx], pcs[val_idx]
    voxels_train, voxels_val = voxels[train_idx], voxels[val_idx]
    cca.fit(voxels_train, pcs_train)
    x_scores_train, y_scores_train = cca.transform(voxels_train, pcs_train)
    x_scores_val, y_scores_val = cca.transform(voxels_val, pcs_val)
    cv_train_r.append(correlation(x_scores_train, y_scores_train))
    cv_val_r.append(correlation(x_scores_val, y_scores_val))

cv_train_r = np.stack(cv_train_r).mean(axis=0)
cv_val_r = np.stack(cv_val_r).mean(axis=0)
print('Cross-validated score correlations\n'
      'Train: Mean={:.3g} Max={:.3g} Min={:.3g}\n'
      'Val: Mean={:.3g} Max={:.3g} Min={:.3g}\n'
      .format(cv_train_r.mean(), cv_train_r.max(), cv_train_r.min(),
              cv_val_r.mean(), cv_val_r.max(), cv_val_r.min()))

x_scores, y_scores = cca.fit_transform(voxels, pcs)
pca_encoder.cpu()
cca_encoder = CCAEncoder(pca_encoder, cca.y_rotations_.astype(np.float32))
save_name = utils.get_run_name('bold5000', 'alexnet', 'conv_3', ['CCA-{}'.format(roi)])
torch.save(cca_encoder, os.path.join('saved_models', save_name + '.pth'))
