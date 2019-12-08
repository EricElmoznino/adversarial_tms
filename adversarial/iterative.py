import torch
from torch.nn import functional as F
import utils


def iterative_perturbation(orig, model, target, eps=0.15, n_iter=50, alpha=0.01, loss_func=F.mse_loss):
    orig = orig.clone()
    target = target.clone()
    orig.requires_grad = True
    perturbation = torch.zeros_like(orig)

    for _ in range(n_iter):
        pred = model(orig + perturbation)
        loss = loss_func(pred, target, reduction='sum')
        loss.backward()

        perturbation += -alpha * torch.sign(orig.grad)
        perturbation = perturbation.clamp(-eps, eps)

        orig.grad.zero_()

    perturbed = orig + perturbation
    perturbed = utils.clamp_imagenet(perturbed)
    perturbed = perturbed.detach()

    errors = performance(orig, perturbed, model, target, loss_func)

    perturbed = perturbed

    return perturbed, errors


def performance(orig, perturbed, model, target, loss_func):
    with torch.no_grad():
        orig_pred = model(orig)
        perturbed_pred = model(perturbed)
    error_orig_target = loss_func(orig_pred, target, reduction='none').sum(dim=-1).cpu().numpy()
    error_perturbed_target = loss_func(perturbed_pred, target, reduction='none').sum(dim=-1).cpu().numpy()
    error_perturbed_orig = loss_func(perturbed_pred, orig_pred, reduction='none').sum(dim=-1).cpu().numpy()
    errors = {'original_to_target': error_orig_target,
              'perturbed_to_target': error_perturbed_target,
              'perturbed_to_original': error_perturbed_orig}
    return errors
torch.nn.MSELoss