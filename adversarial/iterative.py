import torch
from torch.nn import functional as F
import utils


def iterative_perturbation(orig, model, target, eps=0.15, n_iter=50, alpha=0.01, loss_func=F.mse_loss):
    orig = orig.unsqueeze(0)
    target = target.unsqueeze(0)
    orig.requires_grad = True
    perturbation = torch.zeros_like(orig)

    for _ in range(n_iter):
        pred = model(orig + perturbation)
        loss = loss_func(pred, target)
        loss.backward()

        perturbation += -alpha * torch.sign(orig.grad)
        perturbation = perturbation.clamp(-eps, eps)

        orig.grad.zero_()

    perturbed = orig + perturbation
    perturbed = utils.clamp_imagenet(perturbed)
    perturbed = perturbed.detach()

    error_orig, error_perturbed = performance(orig, perturbed, model, target, loss_func)

    perturbed = perturbed.squeeze(0)

    return perturbed, error_orig, error_perturbed


def performance(orig, perturbed, model, target, loss_func):
    with torch.no_grad():
        orig_pred = model(orig)
        perturbed_pred = model(perturbed)
    mse_orig = loss_func(orig_pred, target)
    mse_perturbed = loss_func(perturbed_pred, target)
    return mse_orig, mse_perturbed
