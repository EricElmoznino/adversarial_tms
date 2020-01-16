from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os
from tqdm import tqdm
import numpy as np
import torch
import utils
from disruption import loss_metrics

torch.manual_seed(27)
resolution = 480


def plot_compare_dists(dist1, dist2, name1, name2, title):
    plt.hist(dist1, density=True, alpha=0.5, color='b', label=name1)
    plt.hist(dist2, density=True, alpha=0.5, color='r', label=name2)
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Stimulus disruptions')
    parser.add_argument('--disrupted_folder', required=True, type=str, help='folder containing disrupted images')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    parser.add_argument('--roi', required=True, type=str, choices=['LOC', 'PPA'],
                        help='ROI being targeted for disruption')
    args = parser.parse_args()

    disrupted_stimuli = os.listdir(args.disrupted_folder)
    disrupted_stimuli = [d for d in disrupted_stimuli if '_disrupted.' in d]
    original_stimuli = [d.replace('_disrupted.', '_original.') for d in disrupted_stimuli]

    if args.roi == 'LOC':
        roi_mask = np.array([False for _ in range(100)] + [True for _ in range(100)])
    else:
        roi_mask = np.array([True for _ in range(100)] + [False for _ in range(100)])
    roi_mask = torch.from_numpy(roi_mask.astype(np.uint8))

    encoder = torch.load(os.path.join('saved_models', args.encoder_file))

    on_roi_distances, off_roi_distances = [], []
    for original, disrupted in tqdm(list(zip(original_stimuli, disrupted_stimuli))):
            original = utils.image_to_tensor(os.path.join(args.disrupted_folder, original), resolution=resolution)
            disrupted = utils.image_to_tensor(os.path.join(args.disrupted_folder, disrupted), resolution=resolution)

            with torch.no_grad():
                orig_voxels = encoder(original.unsqueeze(0)).squeeze(0)

            metrics = loss_metrics(original, disrupted, orig_voxels, encoder, roi_mask)
            on_roi_distances.append(metrics['Disrupted to Target (ON ROI)'])
            off_roi_distances.append(metrics['Disrupted to Target (OFF ROI)'])

    off_roi = 'PPA' if args.roi == 'LOC' else 'LOC'
    plot_compare_dists(on_roi_distances, off_roi_distances, args.roi + ' disruption', off_roi + ' disruption',
                       'Black-box adversarial attack (disrupting {})'.format(args.roi))
