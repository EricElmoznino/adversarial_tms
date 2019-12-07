import os
from PIL import Image
import torch
from torchvision.transforms import functional as tr


def listdir(dir):
    files = os.listdir(dir)
    files = [f for f in files if f != '.DS_Store']
    files = sorted(files)
    files = [os.path.join(dir, f) for f in files]
    return files


def image_to_tensor(image_path, resolution=(244, 244)):
    image = Image.open(image_path).convert('RGB')
    image = tr.resize(image, resolution)
    image = tr.to_tensor(image)
    image = tr.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # ImageNet normalization
    return image


def tensor_to_image(image):
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    image = image * std + mean  # ImageNet normalization
    image = tr.to_pil_image(image)
    return image


def clamp_imagenet(image):
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    low = (0 - mean) / std
    high = (1 - mean) / std
    image = torch.max(torch.min(image, high), low)
    return image
