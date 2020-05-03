import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as tr

save_dir = '/home/eric/Desktop/gan_manipulations/bold5000/blended/conv3'
n_samples = 10
generated_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/blended/conv3'
orig_dir = '/home/eric/Documents/datasets/adversarial_tms/blended'
rois = ['loc', 'ppa', 'random']

res = 256
sample_pad = 5
roi_pad = 10

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)


def load_image(image):
    image = Image.open(image).convert('RGB')
    image = tr.resize(image, res)
    if image.width != image.height:
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    return image


def concat_h(img1, img2, pad):
    assert img1.height == img2.height
    new_img = Image.new(img1.mode, (img1.width + pad + img2.width, img1.height))
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
    title = Image.new('RGB', (res * (len(rois) + 1) + roi_pad * len(rois), 50))
    draw_centered_text(title, img_name)
    return title


images = os.listdir(orig_dir)
images = [img for img in images if img != '.DS_Store']
images = sorted(images)

for image_name in images:
    loc = make_roi_title('LOC')
    ppa = make_roi_title('PPA')
    roi_columns = [make_roi_title(roi_name.upper()) for roi_name in rois]
    roi_dirs = [os.path.join(generated_dir, roi_name) for roi_name in rois]
    for sample in range(n_samples):
        sample_name = image_name.replace('.jpg', '_{}.png'.format(sample))
        for i in range(len(roi_columns)):
            image = load_image(os.path.join(roi_dirs[i], sample_name))
            roi_columns[i] = concat_v(roi_columns[i], image, sample_pad)
    orig_column = make_roi_title('ORIG')
    orig_image = load_image(os.path.join(orig_dir, image_name))
    for i in range(n_samples):
        orig_column = concat_v(orig_column, orig_image, sample_pad)
    grid = orig_column
    for i in range(0, len(roi_columns)):
        grid = concat_h(grid, roi_columns[i], roi_pad)
    image_name = image_name.split('.')[0]
    image_title = make_image_title(image_name)
    grid = concat_v(image_title, grid, 0)
    grid.save(os.path.join(save_dir, image_name + '.jpg'))
