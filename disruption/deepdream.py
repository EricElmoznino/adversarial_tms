import torch
from torch.nn import functional as F
import numpy as np
import utils
import scipy.ndimage as nd


def deepdream(orig, model, target, loss_func,
              n_octave, octave_scale,
              n_iter=10, alpha=0.01, max_jitter=32):
    orig = orig.unsqueeze(0)
    target = target.unsqueeze(0)
    if torch.cuda.is_available():
        orig = orig.cuda()
        target = target.cuda()

    # Octave base images
    octaves = [orig]
    orig_size = np.array(orig.shape[-2:])
    for n in range(1, n_octave):
        img = octaves[-1].cpu().numpy()
        img = nd.zoom(img, (1, 1, 1 / octave_scale, 1 / octave_scale), order=1)
        img = torch.from_numpy(img).to(target.device)
        octaves.append(img)

    detail = None
    for octave_base in octaves[::-1]:
        if detail is None:
            detail = torch.zeros_like(octave_base)
        else:
            img = detail.cpu().numpy()
            img = nd.zoom(img, np.array(octave_base.shape) / np.array(detail.shape), order=1)
            detail = torch.from_numpy(img).to(target.device)

        input_octave = octave_base + detail
        output = dream(input_octave, model, target, loss_func,
                       n_iter, alpha, max_jitter)
        detail = output - octave_base

    dreamed = detail + orig
    dreamed = dreamed.squeeze(0)

    return dreamed


def dream(input, model, target, loss_func,
          n_iter=10, alpha=0.01, max_jitter=32):
    for _ in range(n_iter):
        # Jitter shift for prior on correlated nearby pixels
        shift = np.random.randint(-max_jitter, max_jitter + 1, 2)
        input = torch.roll(input, shift.tolist(), [-2, -1])

        # Gradient with respect to input
        input.requires_grad = True
        pred = model(input)
        loss = loss_func(pred, torch.zeros_like(pred))
        loss.backward()

        # Disrupt the image
        mean_grad = input.grad.abs().mean()
        alpha_scaled = alpha / mean_grad
        input = input - alpha_scaled * input.grad

        # Reverse the jitter shift
        input = torch.roll(input, (-shift).tolist(), [-2, -1])

        # Clip image
        input = utils.clamp_imagenet(input)

        # Zero gradients before next pass
        model.zero_grad()

        input = input.detach()

    return input
