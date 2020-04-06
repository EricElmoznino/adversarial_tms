import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as tr


imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if f != '.DS_Store']
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def image_to_tensor(image, resolution=None):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if resolution is not None:
        image = tr.resize(image, resolution)
    if image.width != image.height:
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    image = tr.to_tensor(image)
    image = imagenet_norm(image)
    return image


def tensor_to_image(image):
    image = imagenet_unnorm(image)
    image = tr.to_pil_image(image)
    return image


def imagenet_norm(image):
    dims = len(image.shape)
    if dims < 4:
        image = [image]
    image = [tr.normalize(img, mean=imagenet_mean, std=imagenet_std) for img in image]
    image = torch.stack(image, dim=0)
    if dims < 4:
        image = image.squeeze(0)
    return image


def imagenet_unnorm(image):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    image = image.cpu()
    image = image * std + mean  # ImageNet normalization
    return image


def clamp_imagenet(image):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    mean = mean.to(image.device)
    std = std.to(image.device)
    low = (0 - mean) / std
    high = (1 - mean) / std
    image = torch.max(torch.min(image, high), low)
    return image


def sample_imagenet_noise(resolution):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    noise = torch.distributions.Normal(mean.repeat(1, resolution, resolution),
                                       std.repeat(1, resolution, resolution)).sample()
    noise = imagenet_norm(noise)
    return noise


def get_roi_mask(roi, encoder_file):
    voxel_sizes = {'object2vec': {'LOC': 100, 'PPA': 100},
                   'bold5000': {'LOC': 409, 'PPA': 401, 'EVC': 402, 'OPA': 410, 'RSC': 403}}
    study = encoder_file.split('_')[0].split('=')[-1]
    rois = encoder_file.split('.')[0].split('_')[-1].split('=')[-1].split(',')
    roi_mask = []
    for r in rois:
        roi_mask += [r == roi for _ in range(voxel_sizes[study][r])]
    roi_mask = np.array(roi_mask)
    return roi_mask


def get_run_name(study, feature_extractor, feature_name, rois):
    run_name = '_'.join(['study={}'.format(study),
                         'featextractor={}'.format(feature_extractor),
                         'featname={}'.format(feature_name),
                         'rois={}'.format(','.join(rois))])
    return run_name
