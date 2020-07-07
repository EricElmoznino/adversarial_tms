import os
import shutil
from PIL import Image

save_dir = '/home/eric/Desktop/blended'
generated_dir = '/home/eric/experiments/adversarial_tms/gan_manipulations/bold5000/blended/conv3'
objectdir = '/home/eric/datasets/adversarial_tms/objectcats'
scenedir = '/home/eric/datasets/adversarial_tms/scenecats'
rois = ['loc', 'ppa']

res = 128
cat_pad = 10
sample_pad = 2

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)

images = os.listdir(os.path.join(generated_dir, rois[0]))
images = [img for img in images if img != '.DS_Store']
images = [img for img in images if '.json' not in img]
images = [img for img in images if int(img.split('_')[-1].split('.')[0]) == 0]

source_scenes = os.listdir(scenedir)
source_scenes = [s for s in source_scenes if s != '.DS_Store']
scenes = [img.split('_')[0] for img in images]
scenes = list(set(scenes))

objects = [img.split('_')[1] for img in images]
objects = list(set(objects))

for roi in rois:
    grid = Image.new('RGB', size=(res + cat_pad + len(scenes) * res + (len(objects) - 1) * sample_pad,
                                  res + cat_pad + len(scenes) * res + (len(scenes) - 1) * sample_pad))

    for i, object in enumerate(objects):
        oimg = os.path.join(objectdir, object, object + '001.png')
        oimg = Image.open(oimg).resize((res, res))
        w = i * (res + sample_pad) + res + cat_pad
        grid.paste(oimg, (w, 0))

    for j, scene in enumerate(scenes):
        scene = [s for s in source_scenes if s.split('-')[-1] == scene][0]
        simg = os.path.join(scenedir, scene, scene + '_1.jpg')
        simg = Image.open(simg).resize((res, res))
        h = j * (res + sample_pad) + res + cat_pad
        grid.paste(simg, (0, h))

    for i, object in enumerate(objects):
        for j, scene in enumerate(scenes):
            img = '_'.join([scene, object, '0.jpg'])
            img = os.path.join(generated_dir, roi, img)
            img = Image.open(img).resize((res, res))
            w = i * (res + sample_pad) + res + cat_pad
            h = j * (res + sample_pad) + res + cat_pad
            grid.paste(img, (w, h))

    grid.save(os.path.join(save_dir, roi + '.jpg'))
