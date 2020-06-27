import os
import shutil
import json
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from torchvision.transforms import functional as tr


def make_trial_image(images, radius):
    iw, ih = images[0].width, images[0].height
    width, height = 2 * radius + iw, 2 * radius + ih
    canvas = Image.new(mode='RGB', size=(width, height), color=(255, 255, 255))

    centre = np.array([width / 2, height / 2])
    angle_increment = (2 * np.pi) / len(images)
    if len(images) % 2 == 0:
        start_angle = angle_increment / 2
    else:
        start_angle = 0

    boxes = []
    for i, img in enumerate(images):
        a = start_angle + i * angle_increment
        loc = centre + radius * np.array([np.sin(a), -np.cos(a)])
        loc = loc - np.array([iw / 2, ih / 2])
        box = [int(loc[0]), int(loc[1]), int(loc[0] + iw), int(loc[1] + ih)]
        boxes.append(box)
        canvas.paste(img, box)

    return canvas, boxes


if __name__ == '__main__':
    parser = ArgumentParser(description='Create grid stimuli')
    parser.add_argument('--stimuli_dir', required=True, type=str, help='directory containing stimuli')
    parser.add_argument('--save_dir', required=True, type=str, help='directory to save the trial images')
    parser.add_argument('--radius', default=400, type=int, help='radius of trial centre to image centres')
    parser.add_argument('--img_size', default=256, type=int, help='size of images in trial')
    parser.add_argument('--grayscale', action='store_true', help='make all images grayscale')
    args = parser.parse_args()

    shutil.rmtree(args.save_dir, ignore_errors=True)
    os.mkdir(args.save_dir)
    os.mkdir(os.path.join(args.save_dir, 'images'))

    image_locations = None
    trials = os.listdir(args.stimuli_dir)
    trials = [t for t in trials if t != '.DS_Store']
    for trial in trials:
        trial_dir = os.path.join(args.stimuli_dir, trial)
        images = os.listdir(trial_dir)
        images = [img for img in images if img != '.DS_Store']
        images = [os.path.join(trial_dir, img) for img in images]
        images = sorted(images)
        target_idx = [i for i, img in enumerate(images) if '_target.' in img][0]
        images = [Image.open(img) for img in images]
        images = [tr.resize(img, args.img_size) for img in images]

        if args.grayscale:
            images = [img.convert('L') for img in images]

        trial_image, image_locations = make_trial_image(images, args.radius)

        save_path = os.path.join(args.save_dir, 'images', trial)
        trial_image.save(save_path + '.jpg')
        images[target_idx].save(save_path + '_target.jpg')

    with open(os.path.join(args.save_dir, 'locations.json'), 'w') as f:
        f.write(json.dumps({'locations': image_locations}, indent=2))
