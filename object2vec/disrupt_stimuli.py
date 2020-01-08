from argparse import ArgumentParser
import os
import shutil
import re
import json
from tqdm import tqdm
import numpy as np
import torch
import utils
from object2vec.Subject import Subject
from disruption import deepdream, roi_loss_func, loss_metrics

torch.manual_seed(27)


def load_stimuli(stimuli_folder):
    conditions = os.listdir(stimuli_folder)
    conditions = [c for c in conditions if c != '.DS_Store']
    stimuli = {c: utils.listdir(os.path.join(stimuli_folder, c), path=False) for c in conditions}
    return stimuli


def load_subject(encoder_file):
    subj_num = re.findall('subj=\d\d\d', encoder_file)[0]
    subj_num = int(subj_num[-3:])
    subj = Subject(subj_num)
    return subj


def get_inputs_and_targets(stimuli_folder, stimuli, subject, loss_method, specification_file):
    with open(os.path.join('object2vec', 'specifications', specification_file)) as f:
        specification = json.loads(f.read())
    targets = []
    for orig_cond in specification:
        for stimulus_file in stimuli[orig_cond]:
            stimulus_path = os.path.join(stimuli_folder, orig_cond, stimulus_file)
            stimulus_name = stimulus_file[:-4]
            if loss_method == 'away':
                voxels = subject.condition_voxels[orig_cond]
                targets.append({'orig_cond': orig_cond,
                                'stimulus_path': stimulus_path, 'stimulus_name': stimulus_name,
                                'target_cond': orig_cond, 'target_voxels': voxels})
            elif loss_method == 'towards':
                for target_cond in specification[orig_cond]:
                    voxels = subject.condition_voxels[target_cond]
                    targets.append({'orig_cond': orig_cond,
                                    'stimulus_path': stimulus_path, 'stimulus_name': stimulus_name,
                                    'target_cond': target_cond, 'target_voxels': voxels})
            else:
                raise NotImplementedError('Invalid loss method: {}'.format(loss_method))
    return targets


def save_disrupted_image(save_folder, input_and_target, disrupted_image, metrics):
    def build_folder_structure(sub_path):
        folders = sub_path.split('/')[:-1]
        folders = [os.path.join(*folders[:i+1]) for i in range(len(folders))]
        for f in folders:
            if not os.path.exists(os.path.join(save_folder, f)):
                os.mkdir(os.path.join(save_folder, f))

    def safe_save_image(image, image_sub_path):
        build_folder_structure(image_sub_path)
        image.save(os.path.join(save_folder, image_sub_path))

    def safe_save_file(content, file_sub_path):
        build_folder_structure(file_sub_path)
        with open(os.path.join(save_folder, file_sub_path), 'w') as f:
            f.write(content)

    sub_path = os.path.join(input_and_target['orig_cond'],
                            input_and_target['stimulus_name'],
                            input_and_target['target_cond'])
    safe_save_image(disrupted_image, sub_path + '.png')
    safe_save_file(json.dumps(metrics, indent=2), sub_path + '_metrics.json')


def disrupt_stimuli(save_folder, inputs_and_targets, encoder, roi_mask, target_roi, loss_method):
    roi_mask = torch.from_numpy(roi_mask.astype(np.uint8))

    towards_target = True if loss_method == 'towards' else False
    if not target_roi:
        loss_func = roi_loss_func(None, towards_target)
    else:
        loss_func = roi_loss_func(roi_mask, towards_target)

    for input_and_target in tqdm(inputs_and_targets):
        orig_image = input_and_target['stimulus_path']
        orig_image = utils.image_to_tensor(orig_image)

        target = input_and_target['target_voxels']
        target = torch.from_numpy(target)
        if not target_roi:
            target = target[roi_mask]
        else:
            with torch.no_grad():
                orig_voxels = encoder(orig_image.unsqueeze(0)).squeeze(0)
            target[1 - roi_mask] = orig_voxels[1 - roi_mask]

        disrupted_image = deepdream(orig_image, target, encoder, loss_func)
        metrics = loss_metrics(orig_image, disrupted_image, target, encoder, roi_mask if target_roi else None)

        disrupted_image = utils.tensor_to_image(disrupted_image)
        save_disrupted_image(save_folder, input_and_target, disrupted_image, metrics)


if __name__ == '__main__':
    parser = ArgumentParser(description='Stimulus disruptions')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save disrupted images')
    parser.add_argument('--stimuli_folder', required=True, type=str, help='folder containing stimulus images')
    parser.add_argument('--spec_file', required=True, type=str,
                        help='file specifying which disrupted examples to generate')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    parser.add_argument('--roi', required=True, type=str, choices=['LOC', 'PPA'],
                        help='ROI being targeted for disruption')
    parser.add_argument('--target_roi', action='store_true',
                        help='whether or not to specifically target ROI while leaving other ROIs constant')
    parser.add_argument('--loss_method', type=str, choices=['away', 'towards'], default='away',
                        help='whether to disrupt voxels away from original ones or towards those of another stimulus')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    stimuli = load_stimuli(args.stimuli_folder)
    encoder = torch.load(os.path.join('saved_models', args.encoder_file))
    subject = load_subject(args.encoder_file)
    if not args.target_roi:
        encoder.set_roi_mask(subject.roi_masks[args.roi])

    encoders = {'encoder': encoder, 'random_encoder': encoder.random_weights()}
    inputs_and_targets = get_inputs_and_targets(args.stimuli_folder, stimuli, subject, args.loss_method, args.spec_file)

    for encoder_name in encoders:
        print('Generating disruption examples using: {}'.format(encoder_name))
        save_folder = os.path.join(args.save_folder, encoder_name)
        os.mkdir(save_folder)
        disrupt_stimuli(save_folder, inputs_and_targets, encoders[encoder_name],
                        subject.roi_masks[args.roi], args.target_roi, args.loss_method)
