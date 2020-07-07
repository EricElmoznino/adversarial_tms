import os
import shutil
import math
from PIL import Image
from torchvision.transforms import functional as tr

pc = 'stubbiness'
roi = 'loc'
save_dir = '/home/eric/experiments/adversarial_tms/gan_manipulations/grids/bold5000/brady2008pcs/{}/{}'.format(pc, roi)
orig_dir = '/home/eric/datasets/adversarial_tms/brady2008pcs/{}'.format(pc)
gen_dir = '/home/eric/experiments/adversarial_tms/gan_manipulations/bold5000/brady2008pcs_{}/{}'.format(pc, roi)

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


files = os.listdir(orig_dir)
files = [f for f in files if '.jpg' in f]

neg = [f for f in files if 'neg_' in f]
pos = [f for f in files if 'pos_' in f]

orig_neg = map_to_grid(orig_dir, neg)
orig_pos = map_to_grid(orig_dir, pos)
gen_neg = map_to_grid(gen_dir, [f.split('.')[0] + '_0.jpg' for f in neg])
gen_pos = map_to_grid(gen_dir, [f.split('.')[0] + '_0.jpg' for f in pos])

grid = make_grid([orig_neg, orig_pos, gen_neg, gen_pos], 2, group_pad)
grid.save(os.path.join(save_dir, '{}.jpg'.format(pc)))
