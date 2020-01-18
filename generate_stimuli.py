from argparse import ArgumentParser
import os
import shutil
import json
from tqdm import tqdm
import torch
import utils
from disruption import deepdream, roi_loss_func, loss_metrics

torch.manual_seed(27)
resolution = 375


def generate_stimulus(target, encoder):
    loss_func = roi_loss_func(roi_mask=None, towards_target=True)

    noise = torch.rand(3, resolution, resolution)
    noise = utils.imagenet_norm(noise)

    generated = deepdream(noise, target, encoder, loss_func, n_iter=100)
    metrics = loss_metrics(noise, generated, target, encoder, roi_mask=None)

    return generated, metrics


if __name__ == '__main__':
    parser = ArgumentParser(description='Stimulus disruptions')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save disrupted images')
    parser.add_argument('--targets_folder', required=True, type=str, help='folder containing voxel targets')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    targets = os.listdir(args.targets_folder)
    targets = [t for t in targets if t != '.DS_Store']
    targets = [t for t in targets if '.target.pth' in t]

    encoder = torch.load(os.path.join('saved_models', args.encoder_file))

    print('Generating stimuli')
    for target_name in tqdm(targets):
        target = torch.load(os.path.join(args.targets_folder, target_name))
        generated, metrics = generate_stimulus(target, encoder)

        utils.tensor_to_image(generated).save(os.path.join(args.save_folder, target_name.split('.')[0] + '.png'))
        with open(os.path.join(args.save_folder, target_name.split('.')[0] + '_metrics.json'), 'w') as f:
            f.write(json.dumps(metrics, indent=2))
