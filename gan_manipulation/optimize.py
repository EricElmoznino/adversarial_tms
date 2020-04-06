import torch
from pytorch_pretrained_biggan.utils import truncated_noise_sample
import utils


def optimize(generator, encoder, target, loss_func, initial_latent=None,
             n_iter=200, alpha=2.0, decay=5e-3):
    # Initialization
    if initial_latent is None:
        initial_latent, min_latent, max_latent = latent_params_for_model(generator)
    else:
        _, min_latent, max_latent = latent_params_for_model(generator)
    latent = initial_latent.unsqueeze(0)
    target = target.unsqueeze(0)
    if torch.cuda.is_available():
        latent = latent.cuda()
        target = target.cuda()
        min_latent = min_latent.cuda()
        max_latent = max_latent.cuda()

    lowest_loss = 1000.0
    best_image = None
    best_latent = None

    for i in range(n_iter):
        # Forward pass through generator and encoder
        latent.requires_grad = True
        if type(generator).__name__ == 'BigGAN':
            generated_image = generator(z=latent[:, :128], class_label=latent[:, 128:], truncation=1)
            generated_image = (generated_image + 1) / 2
        else:
            generated_image = generator(latent)
        encoding = encoder(utils.imagenet_norm(generated_image))

        # Backpropagate loss
        loss = loss_func(encoding, target)
        loss.backward()
        if loss.item() < lowest_loss:
            best_image = generated_image
            best_latent = latent
            lowest_loss = loss.item()

        # Gradient update
        step_size = exp_interp(0, n_iter, alpha, 1e-3, 1/3, i)
        latent = update(latent, step_size)

        # Regularization
        latent = torch.max(torch.min(latent, max_latent), min_latent)
        latent = (1 - decay) * latent

        # Zero gradients before next pass
        generator.zero_grad()
        encoder.zero_grad()

    best_image = best_image.squeeze(0).clamp(min=0, max=1).cpu()
    best_latent = best_latent.squeeze(0).cpu()

    return best_image, best_latent, lowest_loss


def update(x, step_size):
    mean_grad = x.grad.abs().mean()
    step_size_scaled = step_size / mean_grad
    x = x - step_size_scaled * x.grad
    x = x.detach()
    return x


def exp_interp(a, b, c, d, k, x):
    x = (x - a) / b
    x = x.clip(min=1e-12)
    e = x ** k
    y = (1 - e) * c + e * d
    return y


def latent_params_for_model(model):
    if type(model).__name__ == 'DeePSiM':
        min_latent = torch.zeros(4096)
        max_latent = torch.load('gan_manipulation/pretrained_models/latent_upper_bounds.pth')
        latent_mean = torch.load('gan_manipulation/pretrained_models/latent_mean.pth')
        latent_std = torch.load('gan_manipulation/pretrained_models/latent_std.pth')
        initial_latent = torch.distributions.Normal(latent_mean, latent_std).sample()
    elif type(model).__name__ == 'BigGAN':
        min_latent = torch.cat([torch.ones(128) * -2, torch.zeros(1000)])
        max_latent = torch.cat([torch.ones(128) * 2, torch.ones(1000)])
        noise = torch.from_numpy(truncated_noise_sample(truncation=1))[0]
        initial_latent = torch.cat([noise, torch.zeros(1000)])
    else:
        raise NotImplementedError('No implementation for model: {}'.format(type(model).__name__))
    return initial_latent, min_latent, max_latent
