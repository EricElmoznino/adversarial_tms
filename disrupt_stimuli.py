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


def disrupt_stimulus(stimulus, encoder, roi_mask):
    roi_mask = torch.from_numpy(roi_mask.astype(np.uint8))
    loss_func = roi_loss_func(roi_mask=roi_mask, towards_target=False)

    with torch.no_grad():
        orig_voxels = encoder(stimulus.unsqueeze(0)).squeeze(0)

    disrupted = deepdream(stimulus, orig_voxels, encoder, loss_func)
    metrics = loss_metrics(stimulus, disrupted, orig_voxels, encoder, roi_mask)

    return disrupted, metrics


if __name__ == '__main__':
    parser = ArgumentParser(description='Stimulus disruptions')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save disrupted images')
    parser.add_argument('--stimuli_folder', required=True, type=str, help='folder containing stimulus images')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    parser.add_argument('--roi', required=True, type=str, choices=['LOC', 'PPA'],
                        help='ROI being targeted for disruption')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    stimuli = os.listdir(args.stimuli_folder)
    stimuli = [s for s in stimuli if s != '.DS_Store']

    if args.roi == 'LOC':
        roi_mask = np.array([False for _ in range(100)] + [True for _ in range(100)])
    else:
        roi_mask = np.array([True for _ in range(100)] + [False for _ in range(100)])

    encoder = torch.load(os.path.join('saved_models', args.encoder_file))
    encoders = {'encoder': encoder, 'random_encoder': encoder.random_weights()}

    for encoder_name in encoders:
        print('Generating disruption examples using: {}'.format(encoder_name))
        save_folder = os.path.join(args.save_folder, encoder_name)
        os.mkdir(save_folder)

        for stimulus_name in tqdm(stimuli):
            stimulus = utils.image_to_tensor(os.path.join(args.stimuli_folder, stimulus_name), resolution)
            disrupted, metrics = disrupt_stimulus(stimulus, encoders[encoder_name], roi_mask)

            shutil.copyfile(os.path.join(args.stimuli_folder, stimulus_name), os.path.join(save_folder, stimulus_name))
            utils.tensor_to_image(disrupted).save(os.path.join(save_folder, stimulus_name.replace('.', '_disrupted.')))
            with open(os.path.join(save_folder, stimulus_name.split('.')[0] + '_metrics.json'), 'w') as f:
                f.write(json.dumps(metrics, indent=2))
