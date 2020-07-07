import math
import numpy as np
import os
import shutil
from PIL import Image
from sklearn.decomposition import PCA
import torch
from torchvision.transforms import functional as tr
from models.feature_extractors import AlexNet
from utils import image_to_tensor


dir = '/home/eric/datasets/adversarial_tms/brady2008'
save_dir = '/home/eric/Desktop/brady2008pcs'
n_pcs = 2
res = 128
sample_pad = 2
group_pad = res * 2
n_exemplars = 40

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)


def load_image(image):
    image = Image.open(image).convert('RGB')
    image = tr.resize(image, 224)
    if image.width != image.height:
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    return image


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


def map_to_grid(images):
    images = make_grid(images, int(math.sqrt(len(images))), sample_pad)
    return images


files = os.listdir(dir)
files = [f for f in files if '.jpg' in f]

model = AlexNet(feature_name='pool')

images = [Image.open(os.path.join(dir, f)).convert('L').convert('RGB') for f in files]
image_tensors = torch.stack([image_to_tensor(img, resolution=224) for img in images])
with torch.no_grad():
    features = model(image_tensors).numpy()

pcs = PCA(n_components=n_pcs).fit_transform(features)
sorted_indices = np.argsort(pcs, axis=0)

low_pcs = [[images[j] for j in sorted_indices[:n_exemplars, i]] for i in range(n_pcs)]
high_pcs = [[images[j] for j in sorted_indices[-n_exemplars:, i]] for i in range(n_pcs)]
for pc in range(n_pcs):
    os.mkdir(os.path.join(save_dir, 'pc_{}'.format(pc)))
    for i, img in enumerate(low_pcs[pc]):
        img.save(os.path.join(save_dir, 'pc_{}'.format(pc), 'neg_{}.jpg'.format(i)))
    for i, img in enumerate(high_pcs[pc]):
        img.save(os.path.join(save_dir, 'pc_{}'.format(pc), 'pos_{}.jpg'.format(i)))

low_pc_grids = [map_to_grid(low_pcs[i]) for i in range(n_pcs)]
high_pc_grids = [map_to_grid(high_pcs[i]) for i in range(n_pcs)]
grids = []
for low_grid, high_grid in zip(low_pc_grids, high_pc_grids):
    grids += [low_grid, high_grid]
make_grid(grids, int(len(grids) / 2), group_pad).show()
