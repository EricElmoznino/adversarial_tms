import os
import shutil
from PIL import Image, ImageDraw, ImageFont

save_dir = '/home/eric/Desktop/roi_optimization_pool'
n_samples = 10
loc_dir = '/home/eric/Documents/experiments/adversarial_tms/optimize_for_loc_pool'
ppa_dir = '/home/eric/Documents/experiments/adversarial_tms/optimize_for_ppa_pool'

res = 256
sample_pad = 5
roi_pad = 10

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)


def concat_h(img1, img2, pad):
    assert img1.height == img2.height
    new_img = Image.new(img1.mode, (img1.width + pad + img1.width, img1.height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width + pad, 0))
    return new_img


def concat_v(img1, img2, pad):
    assert img1.width == img2.width
    new_img = Image.new(img1.mode, (img1.width, img1.height + pad + img2.height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height + pad))
    return new_img


def draw_centered_text(img, msg):
    draw = ImageDraw.Draw(img)
    arial = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 40)
    W, H = img.size
    w, h = arial.getsize(msg)
    draw.text(((W-w)/2,(H-h)/2), msg, font=arial, fill='white')


def make_roi_title(roi):
    title = Image.new('RGB', (res, 50))
    draw_centered_text(title, roi)
    return title


def make_image_title(img_name):
    title = Image.new('RGB', (res * 2 + roi_pad, 50))
    draw_centered_text(title, img_name)
    return title


images = os.listdir(loc_dir)
images = [img for img in images if '.png' in img]
images = sorted(images)
image_samples = {''.join(images[i].split('_')[:-1]): images[i:i+n_samples] for i in range(0, len(images), n_samples)}

for img_name, samples in image_samples.items():
    loc = make_roi_title('LOC')
    ppa = make_roi_title('PPA')
    for image in samples:
        loc_image = Image.open(os.path.join(loc_dir, image)).resize((res, res))
        ppa_image = Image.open(os.path.join(ppa_dir, image)).resize((res, res))
        loc = concat_v(loc, loc_image, sample_pad)
        ppa = concat_v(ppa, ppa_image, sample_pad)
    grid = concat_h(loc, ppa, roi_pad)
    img_title = make_image_title(img_name)
    grid = concat_v(img_title, grid, 0)
    grid.save(os.path.join(save_dir, img_name + '.jpg'))
