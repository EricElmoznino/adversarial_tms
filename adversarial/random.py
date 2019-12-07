import torch
import utils


def random_perturbation(orig, eps):
    perturbation = torch.rand_like(orig)
    perturbation = perturbation / perturbation.norm() * eps
    perturbed = orig + perturbation
    perturbed = utils.clamp_imagenet(perturbed)
    return perturbed
