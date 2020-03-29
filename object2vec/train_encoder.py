from argparse import ArgumentParser
import os
from tqdm import tqdm
import torch
import utils
from models import Encoder, AlexNet, VGG16, RegressionModel
from object2vec.Subject import Subject
from object2vec.regression import cv_regression

resolution = 256


def mean_condition_features(stimuli_folder, model):
    print('Extracting stimuli features')
    conditions = utils.listdir(stimuli_folder)
    condition_features = {}
    for c in tqdm(conditions):
        c_name = c.split('/')[-1]
        stimuli = utils.listdir(c)
        stimuli = [utils.image_to_tensor(s, resolution=resolution) for s in stimuli]
        stimuli = torch.stack(stimuli)
        with torch.no_grad():
            feats = model(stimuli).mean(dim=0)
        condition_features[c_name] = feats
    return condition_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Encoder using object2vec study data')
    parser.add_argument('--stimuli_folder', required=True, type=str, help='folder containing the stimuli images')
    parser.add_argument('--subject_number', default=1, type=int, help='subject number to train encoder for',
                        choices=[1, 2, 3, 4])
    parser.add_argument('--feature_extractor', default='alexnet', type=str, help='feature extraction model')
    parser.add_argument('--feature_name', default='conv_3', type=str, help='feature extraction layer')
    parser.add_argument('--rois', nargs='+', default=['LOC', 'PPA'], type=str, help='ROIs to fit')
    args = parser.parse_args()

    if args.feature_extractor == 'alexnet':
        feat_extractor = AlexNet(args.feature_name)
    elif args.feature_extractor == 'vgg16':
        feat_extractor = VGG16(args.feature_name)
    else:
        raise ValueError('unimplemented feature extractor: {}'.format(args.feature_extractor))

    subject = Subject(args.subject_number, args.rois)
    condition_features = mean_condition_features(args.stimuli_folder, feat_extractor)
    regressor = RegressionModel(condition_features[list(condition_features.keys())[0]].shape[0],
                                subject.n_voxels)

    weight, r = cv_regression(condition_features, subject)
    regressor.set_params(weight)
    print('Mean r score: {:.4f}'.format(r))

    encoder = Encoder(feat_extractor, regressor)
    encoder.eval()
    run_name = utils.get_run_name('object2vec', args.feature_extractor, args.feature_name, args.rois)
    torch.save(encoder, os.path.join('saved_models', run_name + '.pth'))
