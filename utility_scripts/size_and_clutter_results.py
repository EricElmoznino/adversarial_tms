import os
import shutil
import math
from PIL import Image
from torchvision.transforms import functional as tr

save_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/grids/bold5000/size_and_clutter/fc1/ppa'
orig_dir = '/home/eric/Documents/datasets/adversarial_tms/size_and_clutter'
gen_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/size_and_clutter/fc1/ppa'

res = 128
sample_pad = 2
cat_pad = 5
level_pad = 20
orig_gen_pad = res
sample_grid = [4, 3]

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


sizes = [[] for _ in range(6)]
clutters = [[] for _ in range(6)]
img_to_sizeclutter = {}
images = os.listdir(orig_dir)
images = [img for img in images if img != '.DS_Store']
images = sorted(images)
for img in images:
    cat = img.split('_')[0]
    s, c, _ = cat.split('-')
    s, c = int(s), int(c)
    if cat not in img_to_sizeclutter:
        sizes[s - 1].append(cat)
        clutters[c - 1].append(cat)
        img_to_sizeclutter[cat] = {'size': s, 'clutter': c}

# Size image
orig_images, gen_images = [], []
for s, cats in enumerate(sizes):
    s = s + 1
    orig_cat_images, gen_cat_images = [], []
    for cat in cats:
        c = img_to_sizeclutter[cat]['clutter']
        orig_sample_images, gen_sample_images = [], []
        for sample in range(1, sample_grid[0] * sample_grid[1] + 1):
            orig_filename = '{}_{}.jpg'.format(cat, sample)
            gen_filename = '{}_{}_0.jpg'.format(cat, sample)
            orig_path = os.path.join(orig_dir, orig_filename)
            gen_path = os.path.join(gen_dir, gen_filename)
            orig_sample_images.append(load_image(orig_path))
            gen_sample_images.append(load_image(gen_path))
        orig_cat_images.append(make_grid(orig_sample_images, sample_grid[0], sample_pad))
        gen_cat_images.append(make_grid(gen_sample_images, sample_grid[0], sample_pad))
    orig_images.append(make_grid(orig_cat_images, 2, cat_pad))
    gen_images.append(make_grid(gen_cat_images, 2, cat_pad))
orig_images = make_grid(orig_images, 1, level_pad)
gen_images = make_grid(gen_images, 1, level_pad)
size_image = make_grid([orig_images, gen_images], 2, orig_gen_pad)
size_image.save(os.path.join(save_dir, 'size.jpg'))

# Size image summary
orig_images, gen_images = [], []
for s, cats in enumerate(sizes):
    s = s + 1
    orig_cat_images, gen_cat_images = [], []
    for cat in cats:
        c = img_to_sizeclutter[cat]['clutter']
        orig_filename = '{}_{}.jpg'.format(cat, 1)
        gen_filename = '{}_{}_0.jpg'.format(cat, 1)
        orig_path = os.path.join(orig_dir, orig_filename)
        gen_path = os.path.join(gen_dir, gen_filename)
        orig_cat_images.append(load_image(orig_path))
        gen_cat_images.append(load_image(gen_path))
    orig_images.append(make_grid(orig_cat_images, 2, cat_pad))
    gen_images.append(make_grid(gen_cat_images, 2, cat_pad))
orig_images = make_grid(orig_images, 1, level_pad)
gen_images = make_grid(gen_images, 1, level_pad)
size_image = make_grid([orig_images, gen_images], 2, orig_gen_pad)
size_image.save(os.path.join(save_dir, 'size_summary.jpg'))

# Clutter image
orig_images, gen_images = [], []
for c, cats in enumerate(clutters):
    c = c + 1
    orig_cat_images, gen_cat_images = [], []
    for cat in cats:
        s = img_to_sizeclutter[cat]['size']
        orig_sample_images, gen_sample_images = [], []
        for sample in range(1, sample_grid[0] * sample_grid[1] + 1):
            orig_filename = '{}_{}.jpg'.format(cat, sample)
            gen_filename = '{}_{}_0.jpg'.format(cat, sample)
            orig_path = os.path.join(orig_dir, orig_filename)
            gen_path = os.path.join(gen_dir, gen_filename)
            orig_sample_images.append(load_image(orig_path))
            gen_sample_images.append(load_image(gen_path))
        orig_cat_images.append(make_grid(orig_sample_images, sample_grid[0], sample_pad))
        gen_cat_images.append(make_grid(gen_sample_images, sample_grid[0], sample_pad))
    orig_images.append(make_grid(orig_cat_images, 2, cat_pad))
    gen_images.append(make_grid(gen_cat_images, 2, cat_pad))
orig_images = make_grid(orig_images, 1, level_pad)
gen_images = make_grid(gen_images, 1, level_pad)
clutter_image = make_grid([orig_images, gen_images], 2, orig_gen_pad)
clutter_image.save(os.path.join(save_dir, 'clutter.jpg'))

# Clutter image summary
orig_images, gen_images = [], []
for c, cats in enumerate(clutters):
    c = c + 1
    orig_cat_images, gen_cat_images = [], []
    for cat in cats:
        s = img_to_sizeclutter[cat]['size']
        orig_filename = '{}_{}.jpg'.format(cat, 1)
        gen_filename = '{}_{}_0.jpg'.format(cat, 1)
        orig_path = os.path.join(orig_dir, orig_filename)
        gen_path = os.path.join(gen_dir, gen_filename)
        orig_cat_images.append(load_image(orig_path))
        gen_cat_images.append(load_image(gen_path))
    orig_images.append(make_grid(orig_cat_images, 2, cat_pad))
    gen_images.append(make_grid(gen_cat_images, 2, cat_pad))
orig_images = make_grid(orig_images, 1, level_pad)
gen_images = make_grid(gen_images, 1, level_pad)
size_image = make_grid([orig_images, gen_images], 2, orig_gen_pad)
size_image.save(os.path.join(save_dir, 'clutter_summary.jpg'))
