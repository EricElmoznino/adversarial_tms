import os
import shutil
import math
from PIL import Image
from torchvision.transforms import functional as tr
import pandas as pd

save_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/grids/bold5000/greene2009/conv3/ppa'
orig_dir = '/home/eric/Documents/datasets/adversarial_tms/greene2009'
gen_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/greene2009/conv3/ppa'

res = 128
sample_pad = 2
group_pad = res
group_nrows = 5

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)


def load_image(image):
    image = Image.open(image).convert('RGB')
    image = tr.resize(image, res)
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


def map_to_grid(dir, filenames):
    images = [os.path.join(dir, f) for f in filenames]
    images = [Image.open(img) for img in images if os.path.exists(img)]
    images = make_grid(images, group_nrows, sample_pad)
    return images


data = pd.read_csv(os.path.join(orig_dir, 'key.csv'))
cats = [c for c in data.columns if '.Answer' not in c]

for c in cats:
    neg = data[c][data[c + '.Answer'] == 0].tolist()
    pos = data[c][data[c + '.Answer'] == 1].tolist()

    orig_neg = map_to_grid(orig_dir + '/all', neg)
    orig_pos = map_to_grid(orig_dir + '/all', pos)
    gen_neg = map_to_grid(gen_dir, [f.split('.')[0] + '_0.jpg' for f in neg])
    gen_pos = map_to_grid(gen_dir, [f.split('.')[0] + '_0.jpg' for f in pos])

    grid = make_grid([orig_neg, orig_pos, gen_neg, gen_pos], 2, group_pad)
    grid.save(os.path.join(save_dir, '{}.jpg'.format(c)))
