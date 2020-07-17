import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
from models import CCAEncoder
import utils

n_components = 100
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

assert (pcs.mean(axis=0) < 1e-5).all()
cca = CCA(n_components=n_components, scale=False)
x_scores, y_scores = cca.fit_transform(voxels, pcs)
r = correlation(x_scores, y_scores)
print('Score correlations\nMean: {}\nMax: {}\nMin: {}'.format(r.mean(), r.max(), r.min()))

pca_encoder.cpu()
cca_encoder = CCAEncoder(pca_encoder, cca.y_rotations_.astype(np.float32))
torch.save(cca_encoder, os.path.join('saved_models', pca_encoder_name.replace('PCA.pth', 'CCA-{}.pth'.format(roi))))
