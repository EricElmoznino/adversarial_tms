from argparse import ArgumentParser
import os
import shutil
import json
from tqdm import tqdm
import numpy as np
import torch
import utils
from disruption import deepdream, roi_loss_func, loss_metrics

torch.manual_seed(27)
resolution = 480


def disrupt_stimulus(stimulus, target, encoder, roi_mask, towards_target):
    roi_mask = torch.from_numpy(roi_mask.astype(np.uint8))
    loss_func = roi_loss_func(roi_mask, towards_target)

    with torch.no_grad():
        orig_voxels = encoder(stimulus.unsqueeze(0)).squeeze(0)
    target[1 - roi_mask] = orig_voxels[1 - roi_mask]

    disrupted = deepdream(stimulus, target, encoder, loss_func, n_octave=1)
    metrics = loss_metrics(stimulus, disrupted, target, encoder, roi_mask)

    return disrupted, metrics


if __name__ == '__main__':
    parser = ArgumentParser(description='Stimulus disruptions')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save disrupted images')
    parser.add_argument('--stimuli_folder', required=True, type=str, help='folder containing stimulus images')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    parser.add_argument('--roi', required=True, type=str, choices=['LOC', 'PPA'],
                        help='ROI being targeted for disruption')
    parser.add_argument('--towards_target', action='store_true',
                        help='whether to disrupt voxels away from target or towards it')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    stimuli = os.listdir(args.stimuli_folder)
    stimuli = [s for s in stimuli if s != '.DS_Store']
    stimuli = [s for s in stimuli if '.target.pth' not in s]

    if args.roi == 'LOC':
        roi_mask = np.array([False for _ in range(100)] + [True for _ in range(100)])
    else:
        roi_mask = np.array([True for _ in range(100)] + [False for _ in range(100)])

    encoder = torch.load(os.path.join('saved_models', args.encoder_file))

    print('Generating disruption examples')
    for stimulus_name in tqdm(stimuli):
        stimulus = utils.image_to_tensor(os.path.join(args.stimuli_folder, stimulus_name), resolution)
        target = torch.load(os.path.join(args.stimuli_folder, stimulus_name + '.target.pth'))
        disrupted, metrics = disrupt_stimulus(stimulus, target, encoder, roi_mask, args.towards_target)

        shutil.copyfile(os.path.join(args.stimuli_folder, stimulus_name),
                        os.path.join(args.save_folder, stimulus_name).replace('.', '_original.'))
        utils.tensor_to_image(disrupted).save(os.path.join(args.save_folder, stimulus_name.replace('.', '_disrupted.')))
        with open(os.path.join(args.save_folder, stimulus_name.split('.')[0] + '_metrics.json'), 'w') as f:
            f.write(json.dumps(metrics, indent=2))
