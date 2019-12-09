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
from adversarial import iterative_perturbation

torch.manual_seed(27)
batch_size = 32


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


def get_adversarial_inputs(stimuli_folder, stimuli, subject, specification_file):
    with open(os.path.join('object2vec', 'perturb_specifications', specification_file)) as f:
        specification = json.loads(f.read())
    targets = []
    for orig_cond in specification:
        for stimulus_file in stimuli[orig_cond]:
            stimulus_path = os.path.join(stimuli_folder, orig_cond, stimulus_file)
            stimulus_name = stimulus_file[:-4]
            for adv_cond in specification[orig_cond]:
                voxels = subject.condition_voxels[adv_cond]
                targets.append({'orig_cond': orig_cond, 'stimulus_path': stimulus_path, 'stimulus_name': stimulus_name,
                                'adv_cond': adv_cond, 'adv_voxels': voxels})
    return targets


def save_adversarial_batch(save_folder, batch, perturbed_images, errors):
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

    for input, perturb, error in zip(batch, perturbed_images, errors):
        sub_path = os.path.join(input['orig_cond'], input['stimulus_name'], input['adv_cond'])
        safe_save_image(perturb, sub_path + '.png')
        safe_save_file(json.dumps(error), sub_path + '_errors.json')


def make_adversarial_examples(save_folder, inputs, encoder, roi_mask, targeted):
    roi_mask = torch.from_numpy(roi_mask.astype(np.uint8))

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i + batch_size]

        orig_images = [b['stimulus_path'] for b in batch]
        orig_images = [utils.image_to_tensor(img) for img in orig_images]
        orig_images = torch.stack(orig_images)

        targets = [b['adv_voxels'] for b in batch]
        targets = [torch.from_numpy(t) for t in targets]
        targets = torch.stack(targets)
        if not targeted:
            targets = targets[:, roi_mask]
        else:
            with torch.no_grad():
                orig_voxels = encoder(orig_images)
            targets[:, 1 - roi_mask] = orig_voxels[:, 1 - roi_mask]

        perturbed_images, errors = iterative_perturbation(orig_images, encoder, targets)
        perturbed_images = [utils.tensor_to_image(img) for img in perturbed_images]
        save_adversarial_batch(save_folder, batch, perturbed_images, errors)


if __name__ == '__main__':
    parser = ArgumentParser(description='Adversarial perturbations')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save adversarial examples')
    parser.add_argument('--stimuli_folder', required=True, type=str, help='folder containing the stimuli images')
    parser.add_argument('--specification_file', required=True, type=str,
                        help='file specifying which adversarial examples to generate')
    parser.add_argument('--roi', required=True, type=str, choices=['LOC', 'PPA'],
                        help='roi to generate advesarial examples for')
    parser.add_argument('--targeted', action='store_true',
                        help='whether or not to specifically target roi by leaving other rois constant')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    stimuli = load_stimuli(args.stimuli_folder)
    encoder = torch.load(os.path.join('saved_models', args.encoder_file))
    subject = load_subject(args.encoder_file)
    if not args.targeted:
        encoder.set_roi_mask(subject.roi_masks[args.roi])

    encoders = {'encoder': encoder, 'random_encoder': encoder.random_weights()}
    adversarial_inputs = get_adversarial_inputs(args.stimuli_folder, stimuli, subject, args.specification_file)

    for encoder_name in encoders:
        print('Generating adversarial examples using: {}'.format(encoder_name))
        save_folder = os.path.join(args.save_folder, encoder_name)
        os.mkdir(save_folder)
        make_adversarial_examples(save_folder, adversarial_inputs, encoders[encoder_name],
                                  subject.roi_masks[args.roi], args.targeted)
