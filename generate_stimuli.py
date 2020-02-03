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


def generate_stimulus(target, encoder, towards_target):
    loss_func = roi_loss_func(roi_mask=None, towards_target=towards_target)

    noise = torch.rand(3, resolution, resolution)
    noise = utils.imagenet_norm(noise)

    generated = deepdream(noise, target, encoder, loss_func, n_iter=20)
    metrics = loss_metrics(noise, generated, target, encoder, roi_mask=None)

    return generated, metrics


if __name__ == '__main__':
    parser = ArgumentParser(description='Stimulus disruptions')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save disrupted images')
    parser.add_argument('--targets_folder', default=None, type=str,
                        help='folder containing voxel targets (if not provided, activation will be maximized)')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    encoder = torch.load(os.path.join('saved_models', args.encoder_file))

    if args.targets_folder is not None:
        print('Generating targeted stimuli')
        targets = os.listdir(args.targets_folder)
        targets = [t for t in targets if t != '.DS_Store']
        targets = [t for t in targets if '.target.pth' in t]
        for target_name in tqdm(targets):
            target = torch.load(os.path.join(args.targets_folder, target_name))
            generated, metrics = generate_stimulus(target, encoder, towards_target=True)
            utils.tensor_to_image(generated).save(os.path.join(args.save_folder, target_name.split('.')[0] + '.png'))
            with open(os.path.join(args.save_folder, target_name.split('.')[0] + '_metrics.json'), 'w') as f:
                f.write(json.dumps(metrics, indent=2))
    else:
        print('Generating untargeted stimuli')
        for i in tqdm(range(20)):
            target = torch.zeros(encoder.regressor.linear.out_features)
            generated, metrics = generate_stimulus(target, encoder, towards_target=False)
            utils.tensor_to_image(generated).save(os.path.join(args.save_folder, '{:05d}.png'.format(i)))
            with open(os.path.join(args.save_folder, '{:05d}_metrics.json'.format(i)), 'w') as f:
                f.write(json.dumps(metrics, indent=2))
