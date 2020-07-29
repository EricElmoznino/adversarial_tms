import os
import shutil
import math
from PIL import Image
from torchvision.transforms import functional as tr

rois = ['loc', 'ppa', 'cca-loc', 'cca-ppa', 'cca-loc-allsubj', 'cca-ppa-allsubj', 'pca']
save_dir = '/home/eric/experiments/adversarial_tms/gan_manipulations/grids/bold5000/bao2020'
orig_dir = '/home/eric/datasets/adversarial_tms/bao2020'
gen_dir = '/home/eric/experiments/adversarial_tms/gan_manipulations/bold5000/bao2020'

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
    images = [load_image(img) for img in images]
    images = make_grid(images, group_nrows, sample_pad)
    return images


files = os.listdir(orig_dir)
files = [f for f in files if '.jpg' in f]

for roi in rois:
    for neg, pos in [('stubby', 'spiky'), ('inanimate', 'animate')]:
        neg_files = [f for f in files if neg in f.split('.')[0].split('_')]
        pos_files = [f for f in files if pos in f.split('.')[0].split('_')]

        orig_neg = map_to_grid(orig_dir, neg_files)
        orig_pos = map_to_grid(orig_dir, pos_files)
        gen_neg = map_to_grid(os.path.join(gen_dir, roi),
                              [f.split('.')[0] + '_0.jpg' for f in neg_files])
        gen_pos = map_to_grid(os.path.join(gen_dir, roi),
                              [f.split('.')[0] + '_0.jpg' for f in pos_files])

        grid = make_grid([orig_neg, orig_pos, gen_neg, gen_pos], 2, group_pad)
        grid.save(os.path.join(save_dir, '{}_{}.jpg'.format(pos, roi)))
