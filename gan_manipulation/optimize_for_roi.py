from argparse import ArgumentParser
import os
import shutil
import json
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_pil_image
from pytorch_pretrained_biggan import BigGAN
from gan_manipulation import DeePSiM
from gan_manipulation import optimize
from disruption import roi_loss_func


def generate_samples(save_path, generator, encoder, target, loss_func, n_samples):
    mean_loss = 0
    for i in range(n_samples):
        generated_image, _, loss, _ = optimize(generator, encoder, target, loss_func)
        generated_image = to_pil_image(generated_image)
        generated_image.save('{}_{}.jpg'.format(save_path, i))
        mean_loss += loss / n_samples
    with open('{}_metrics.json'.format(save_path), 'w') as f:
        f.write(json.dumps({'mean_final_loss': mean_loss}, indent=2))


if __name__ == '__main__':
    parser = ArgumentParser(description='Optimize an image to maximize a class probability using a GAN')
    parser.add_argument('--save_folder', required=True, type=str, help='folder to save generated images')
    parser.add_argument('--encoder_file', required=True, type=str, help='name of the encoder file')
    parser.add_argument('--targets_folder', default=None, type=str,
                        help='folder containing voxel targets (if not provided, activation will be maximized)')
    parser.add_argument('--n_samples', default=1, type=int, help='number of samples to generate per target')
    parser.add_argument('--model', default='deepsim', type=str, choices=['deepsim', 'biggan'],
                        help='which generator model to use for optimizing images')
    args = parser.parse_args()

    shutil.rmtree(args.save_folder, ignore_errors=True)
    os.mkdir(args.save_folder)

    encoder = torch.load(os.path.join('saved_models', args.encoder_file),
                         map_location=lambda storage, loc: storage)
    if args.model == 'deepsim':
        generator = DeePSiM()
    elif args.model == 'biggan':
        generator = BigGAN.from_pretrained('biggan-deep-256')
    if torch.cuda.is_available():
        encoder.cuda()
        generator.cuda()

    if args.targets_folder is not None:
        print('Generating targeted stimuli')
        loss_func = roi_loss_func(roi_mask=None, towards_target=True)
        targets = os.listdir(args.targets_folder)
        targets = [t for t in targets if t != '.DS_Store']
        targets = [t for t in targets if '.pth' in t]
        for target_name in tqdm(targets):
            target = torch.load(os.path.join(args.targets_folder, target_name))
            save_path = os.path.join(args.save_folder, target_name.split('.')[0])
            generate_samples(save_path, generator, encoder, target, loss_func, args.n_samples)
    else:
        print('Generating untargeted stimuli')
        loss_func = roi_loss_func(roi_mask=None, towards_target=False)
        target = torch.zeros(encoder.regressor.linear.out_features)
        save_path = os.path.join(args.save_folder, 'sample')
        generate_samples(save_path, generator, encoder, target, loss_func, args.n_samples)
