from argparse import ArgumentParser
import os
import shutil
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
from pytorch_pretrained_biggan import BigGAN
from gan_manipulation import DeePSiM
from gan_manipulation import optimize
from torchvision.models import alexnet
from utils import image_to_tensor


def get_loss_func_for_class(c):
    def loss_func(input, target):
        return -input[:, c].sum()
    return loss_func


if __name__ == '__main__':
    parser = ArgumentParser(description='Optimize an image to maximize a class probability using a GAN')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save generated images')
    parser.add_argument('--image_folder', required=True, type=str, help='images to generate for')
    parser.add_argument('--model', default='deepsim', type=str, choices=['deepsim', 'biggan'],
                        help='which generator model to use for optimizing images')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    encoder = alexnet(pretrained=True)
    encoder.classifier = encoder.classifier[:-1]
    encoder.eval()
    if args.model == 'deepsim':
        generator = DeePSiM()
    elif args.model == 'biggan':
        generator = BigGAN.from_pretrained('biggan-deep-256')
    if torch.cuda.is_available():
        encoder.cuda()
        generator.cuda()

    for image_file in tqdm(os.listdir(args.image_folder)):
        image = image_to_tensor(os.path.join(args.image_folder, image_file), resolution=256)
        with torch.no_grad():
            target = encoder(image.unsqueeze(0)).squeeze(0)
            target_mean_square = (target ** 2).mean().item()
        generated_image, _, lowest_loss = optimize(generator, encoder, target, F.mse_loss)
        generated_image = to_pil_image(generated_image)
        generated_image.save(os.path.join(args.save_folder, image_file))
        print('Lowest loss for {}:\t{}\nMean square of target for {}:\t{}\n'
              .format(image_file, lowest_loss, image_file, target_mean_square))
