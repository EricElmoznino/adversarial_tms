import os
import shutil
import math
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from gan_manipulation import DeePSiM
from gan_manipulation import optimize
from disruption import roi_loss_func
from utils import image_to_tensor
from matplotlib import pyplot as plt

alphas = [0.01, 0.1, 1, 2, 3]
decays = [5e-3]
image_path = '/home/eelmozn1/datasets/adversarial_tms/scenecats/3-3-japaneseroom/3-3-japaneseroom_9.jpg'
save_folder = '/home/eelmozn1/Desktop/parameter_sweep'
encoder_file = 'study=bold5000_featextractor=alexnet_featname=conv_3_rois=PPA.pth'
generator = DeePSiM()
encoder = torch.load(os.path.join('saved_models', encoder_file),
                     map_location=lambda storage, loc: storage)
if torch.cuda.is_available():
    encoder.cuda()
    generator.cuda()

shutil.rmtree(save_folder, ignore_errors=True)
os.mkdir(save_folder)

shutil.copyfile(image_path, os.path.join(save_folder, 'original.jpg'))

image = image_to_tensor(image_path, resolution=256)
with torch.no_grad():
    if torch.cuda.is_available():
        target = encoder(image.unsqueeze(0).cuda()).squeeze(0).cpu()
    else:
        target = encoder(image.unsqueeze(0)).squeeze(0)

loss_func = roi_loss_func(roi_mask=None, towards_target=True)

gen_images = []
fig, axs = plt.subplots(len(alphas), len(decays), squeeze=False, figsize=(len(decays) * 10, len(alphas) * 5))
for j, decay in enumerate(decays):
    for i, alpha in tqdm(enumerate(alphas)):
        gen_image, _, loss, losses = optimize(generator, encoder, target, loss_func,
                                              alpha=alpha, decay=decay)
        gen_images.append(to_pil_image(gen_image))

        axs[i, j].plot(range(len(losses)), losses)
        axs[i, j].set_title('alpha: {:.3g}, decay: {:.3g}, min_loss: {:.0f}'.format(alpha, decay, loss))
        axs[i, j].set_xlabel('Iteration')
        axs[i, j].set_ylabel('Loss')


def make_grid(imgs, n_rows, pad):
    assert len(imgs) > 0
    n_cols = math.ceil(len(imgs) / n_rows)
    w, h = imgs[0].width, imgs[0].height
    grid = Image.new(imgs[0].mode, (w * n_cols + pad * (n_cols - 1),
                                    h * n_rows + pad * (n_rows - 1)),
                     color=(256, 256, 256))
    for i, img in enumerate(imgs):
        row = int(i / n_cols)
        col = i % n_cols
        grid.paste(img, ((w + pad) * col, (h + pad) * row))
    return grid


gen_summary = make_grid(gen_images, len(alphas), 5)
gen_summary.save(os.path.join(save_folder, 'generated.jpg'))

fig.tight_layout()
plt.savefig(os.path.join(save_folder, 'loss_decay.jpg'))
plt.close()
