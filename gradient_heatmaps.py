from argparse import ArgumentParser
import os
import shutil
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as tr
import utils

resolution = 480


def make_heatmaps(stimulus, loc_grad, ppa_grad):
    stimulus = utils.imagenet_unnorm(stimulus)

    loc = loc_grad.abs()
    loc = loc.sum(dim=0)
    loc = loc / loc.max()
    loc = torch.stack((loc, torch.zeros_like(loc), torch.zeros_like(loc)))
    loc_heatmap = 0.2 * stimulus + 0.8 * loc
    loc_heatmap = tr.to_pil_image(loc_heatmap)

    ppa = ppa_grad.abs()
    ppa = ppa.sum(dim=0)
    ppa = ppa / ppa.max()
    ppa = torch.stack((torch.zeros_like(ppa), torch.zeros_like(ppa), ppa))
    ppa_heatmap = 0.2 * stimulus + 0.8 * ppa
    ppa_heatmap = tr.to_pil_image(ppa_heatmap)

    dif = loc_grad.abs() - ppa_grad.abs()
    dif = dif.sum(dim=0)
    dif = dif / dif.abs().max()
    loc_selective = dif.clamp(min=0)
    ppa_selective = -dif.clamp(max=0)
    pixel_selectiveness = torch.stack((loc_selective, torch.zeros_like(loc_selective), ppa_selective))
    selectivity_heatmap = 0.2 * stimulus + 0.8 * pixel_selectiveness
    selectivity_heatmap = tr.to_pil_image(selectivity_heatmap)

    return selectivity_heatmap, loc_heatmap, ppa_heatmap


if __name__ == '__main__':
    parser = ArgumentParser(description='Stimulus disruptions')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save gradient heatmaps')
    parser.add_argument('--stimuli_folder', required=True, type=str, help='folder containing stimulus images')
    parser.add_argument('--encoder_file', required=True, type=str, help='path to the encoder file')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    stimuli = os.listdir(args.stimuli_folder)
    stimuli = [s for s in stimuli if s != '.DS_Store']

    encoder = torch.load(os.path.join('saved_models', args.encoder_file))

    loc_mask = torch.tensor([0 for _ in range(100)] + [1 for _ in range(100)], dtype=torch.uint8)
    ppa_mask = torch.tensor([1 for _ in range(100)] + [0 for _ in range(100)], dtype=torch.uint8)

    for stimulus_name in tqdm(stimuli):
        stimulus = utils.image_to_tensor(os.path.join(args.stimuli_folder, stimulus_name), resolution).unsqueeze(0)
        stimulus.requires_grad = True

        voxels = encoder(stimulus)
        loc_voxels = voxels[:, loc_mask]
        loc_voxels_norm = F.mse_loss(loc_voxels, torch.zeros_like(loc_voxels))
        loc_voxels_norm.backward(retain_graph=True)
        loc_grad = stimulus.grad

        encoder.zero_grad()
        stimulus.grad = None

        ppa_voxels = voxels[:, ppa_mask]
        ppa_voxels_norm = F.mse_loss(ppa_voxels, torch.zeros_like(ppa_voxels))
        ppa_voxels_norm.backward()
        ppa_grad = stimulus.grad

        selective, loc, ppa = make_heatmaps(stimulus.squeeze(0), loc_grad.squeeze(0), ppa_grad.squeeze(0))

        selective.save(os.path.join(args.save_folder, stimulus_name.replace('.', '_SELECTIVE.')))
        loc.save(os.path.join(args.save_folder, stimulus_name.replace('.', '_LOC.')))
        ppa.save(os.path.join(args.save_folder, stimulus_name.replace('.', '_PPA.')))

        encoder.zero_grad()
