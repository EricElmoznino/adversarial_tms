from argparse import ArgumentParser
import os
import shutil
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_pil_image
from pytorch_pretrained_biggan import BigGAN
from gan_manipulation import DeePSiM
from gan_manipulation import optimize
from torchvision.models import alexnet


def get_loss_func_for_class(c):
    def loss_func(input, target):
        return -input[:, c].sum()
    return loss_func


if __name__ == '__main__':
    parser = ArgumentParser(description='Optimize an image to maximize a class probability using a GAN')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save generated images')
    parser.add_argument('--classes', nargs='+', default=[629], type=int, help='classes to generate')
    parser.add_argument('--model', default='deepsim', type=str, choices=['deepsim', 'biggan'],
                        help='which generator model to use for optimizing images')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    encoder = alexnet(pretrained=True)
    encoder.eval()
    if args.model == 'deepsim':
        generator = DeePSiM()
    elif args.model == 'biggan':
        generator = BigGAN.from_pretrained('biggan-deep-256')
    if torch.cuda.is_available():
        encoder.cuda()
        generator.cuda()

    for c in tqdm(args.classes):
        generated_image, _, best_act, _ = optimize(generator, encoder, torch.zeros(1), get_loss_func_for_class(c))
        generated_image = to_pil_image(generated_image)
        generated_image.save(os.path.join(args.save_folder, 'class={:05d}.jpg'.format(c)))
        best_act = -best_act
        print('Highest activation for unit {}: {:.3f}'.format(c, best_act))
