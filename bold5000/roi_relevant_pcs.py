import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import utils

n_pcs = 10
roi = 'PPA'
pca_encoder_name = 'study=bold5000_featextractor=alexnet_featname=conv_3_rois=PCA.pth'
bold5000_folder = '/home/eelmozn1/datasets/adversarial_tms/bold5000'


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


pca_encoder = torch.load(os.path.join('saved_models', pca_encoder_name), map_location=lambda storage, loc: storage)
if torch.cuda.is_available():
    pca_encoder.cuda()

voxels = voxel_data(os.path.join(bold5000_folder, 'subj1.npy'), roi)
pcs = condition_features(os.path.join(bold5000_folder, 'stimuli'), pca_encoder)
pcs = np.stack([pcs[c] for c in voxels], axis=0)
voxels = np.stack([voxels[c] for c in voxels], axis=0)

regr = LinearRegression()
pred_pcs = cross_val_predict(regr, voxels, pcs, cv=5)
r = correlation(pred_pcs, pcs)

r_order = np.argsort(r)[::-1][:n_pcs]
print('{} most relevant PCs: \n{}PC prediction r: {}'.format(n_pcs, r_order, r[r_order]))

pca_encoder.cpu()
new_projection = nn.Linear(pca_encoder.projection.in_features, n_pcs, bias=False)
new_projection.weight.data = pca_encoder.projection.weight.data[r_order, :]
pca_encoder.projection = new_projection
pca_encoder.eval()
torch.save(pca_encoder, os.path.join('saved_models', pca_encoder_name.replace('.pth', '_{}-{}.pth'.format(roi, n_pcs))))
