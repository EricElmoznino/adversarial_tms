import math
import torch
from torch import nn
import utils


def deepvisualize(orig, target, model, loss_func,
                  alpha=0.1, n_iter=50, decay=1e-4, blur_std=1.0, blur_freq=4):
    disrupted = orig.unsqueeze(0)
    target = target.unsqueeze(0)
    if torch.cuda.is_available():
        disrupted = disrupted.cuda()
        target = target.cuda()

    if blur_freq > 0 and blur_std > 0:
        blur = get_gaussian_kernel(sigma=blur_std)

    for i in range(n_iter):
        # Gradient with respect to input
        disrupted.requires_grad = True
        pred = model(disrupted)
        loss = loss_func(pred, target)
        loss.backward()

        # Disrupt the image
        mean_grad = disrupted.grad.abs().mean()
        alpha_scaled = alpha / mean_grad
        disrupted = disrupted - alpha_scaled * disrupted.grad

        # L2 regularization
        disrupted = (1 - decay) * disrupted

        # Gaussian filter
        if blur_freq > 0 and blur_std > 0:
            if blur_std < .3:
                print('Warning: blur-radius of .3 or less works very poorly')
            if i % blur_freq == 0:
                with torch.no_grad():
                    disrupted = blur(disrupted)

        # Clip image
        disrupted = utils.clamp_imagenet(disrupted)

        # Zero gradients before next pass
        model.zero_grad()

        disrupted = disrupted.detach()

    disrupted = disrupted.squeeze(0).cpu()

    return disrupted


def get_gaussian_kernel(sigma=1.0, truncate=4.0, channels=3):
    kernel_size = int((truncate * sigma + 0.5) * 2)

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    pad = nn.ReflectionPad2d(kernel_size // 2)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return nn.Sequential(pad, gaussian_filter)
