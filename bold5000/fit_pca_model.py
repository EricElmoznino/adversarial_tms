from argparse import ArgumentParser
import os
import random
import numpy as np
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA
import utils
from models import PCAEncoder, AlexNet, VGG16

random.seed(27)
resolution = 256


def condition_features(stimuli_folder, model):
    print('Extracting features')
    stimuli = os.listdir(stimuli_folder)
    condition_features = []
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
        condition_features.append(batch_feats)
    condition_features = np.concatenate(condition_features)
    return condition_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Encoder using object2vec study data')
    parser.add_argument('--bold5000_folder', required=True, type=str, help='folder containing the stimuli images')
    parser.add_argument('--feature_extractor', default='alexnet', type=str, help='feature extraction model')
    parser.add_argument('--feature_name', default='conv_3', type=str, help='feature extraction layer')
    parser.add_argument('--n_pcs', default=1200, type=int, help='number of pcs to fit')
    args = parser.parse_args()

    if args.feature_extractor == 'alexnet':
        feat_extractor = AlexNet(args.feature_name)
    elif args.feature_extractor == 'vgg16':
        feat_extractor = VGG16(args.feature_name)
    else:
        raise ValueError('unimplemented feature extractor: {}'.format(args.feature_extractor))
    if torch.cuda.is_available():
        feat_extractor.cuda()

    features = condition_features(os.path.join(args.bold5000_folder, 'stimuli'), feat_extractor)

    pca = PCA(n_components=args.n_pcs)
    pca.fit(features)
    print('\nMean Explained Variance: {:.4f}'.format(np.mean(pca.explained_variance_ratio_.mean())))

    encoder = PCAEncoder(feat_extractor, pcs=pca.components_, mean=pca.mean_)
    encoder.eval()
    run_name = utils.get_run_name('bold5000', args.feature_extractor, args.feature_name, ['PCA'])
    torch.save(encoder, os.path.join('saved_models', run_name + '.pth'))
