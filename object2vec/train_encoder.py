from argparse import ArgumentParser
import os
import torch
from helpers import utils
from models import AlexNet, RegressionModel
from object2vec.Subject import Subject


def mean_condition_features(stimuli_folder, model):
    conditions = utils.listdir(stimuli_folder)
    condition_features = {}
    for c in conditions:
        c_name = c.split('/')[-1]
        stimuli = utils.listdir(c)
        stimuli = [utils.image_to_tensor(s) for s in stimuli]
        stimuli = torch.stack(stimuli)
        feats = model(stimuli).mean(dim=0)
        condition_features[c_name] = feats
    return condition_features


if __name__ == '__main__':
    parser = ArgumentParser(description='Encoder using object2vec study data')
    parser.add_argument('--stimuli_folder', required=True, type=str, help='folder containing the stimuli images')
    parser.add_argument('--subject_number', default=1, type=int, help='subject number to train encoder for',
                        choices=[1, 2, 3, 4])
    parser.add_argument('--feature_extractor', default='alexnet', type=str, help='feature extraction model')
    parser.add_argument('--feature_names', nargs='+', default='pool', type=str, help='feature extraction layers')
    args = parser.parse_args()

    run_name = '_'.join(['study=object2vec',
                         'subj={:03}'.format(args.subject_number),
                         'featextractor={}'.format(args.feature_extractor),
                         'featnames={}'.format(','.join(args.feature_names))])

    if args.feature_extractor == 'alexnet':
        feat_extractor = AlexNet(args.feature_names)
    else:
        raise ValueError('unimplemented feature extractor: {}'.format(args.feature_extractor))

    condition_features = mean_condition_features(args.stimuli_folder, feat_extractor)
    subject = Subject('object2vec/subject_data/subj{:03}'.format(args.subject_number))
    encoder = RegressionModel(condition_features[list(condition_features.keys())[0]].shape[0],
                              subject.n_voxels)
