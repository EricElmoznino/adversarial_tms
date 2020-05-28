import os
import shutil
from copy import copy
import json
import random

random.seed(27)

orig_dir = {'scene': '/home/eric/Documents/datasets/adversarial_tms/mturkscenes/',
            'object': '/home/eric/Documents/datasets/adversarial_tms/mturkobjects/'}
generated_dir = {'scene': '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/mturkscenes/conv3/',
                 'object': '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/mturkobjects/conv3/'}
save_dir = 'web/assets/'

use_same_img = False
n_catch = 10
n_per_cat = 9
n_samples = 10
rois = ['ppa', 'loc', 'evc', 'random']

train_cats = [('mountain', 'scene'), ('boathouse', 'scene'), ('store', 'scene'),
              ('flower', 'object'), ('car', 'object'), ('chair', 'object')]
exp_cats = [('beach', 'scene'), ('street', 'scene'), ('kitchen', 'scene'),
            ('forest', 'scene'), ('building', 'scene'), ('office', 'scene'),
            ('bird', 'object'), ('basketball', 'object'), ('mouse', 'object'),
            ('apple', 'object'), ('helmet', 'object'), ('mug', 'object')]
n_per_train = 2
n_per_exp = 4

img_dir = save_dir + 'images/'
trial_data_path = save_dir + 'trialData.json'
shutil.rmtree(img_dir, ignore_errors=True)
os.mkdir(img_dir)
shutil.rmtree(trial_data_path, ignore_errors=True)

trial_data = {'images': [], 'trainTrials': [], 'experimentTrials': []}

for cat, cond in train_cats:
    for orig_idx in range(n_per_train):
        trial = {'condition': cond}

        orig_name = '{}_{:03d}.jpg'.format(cat, orig_idx)
        shutil.copy(orig_dir[cond] + orig_name, img_dir + orig_name)
        trial['orig'] = orig_name
        trial_data['images'].append(orig_name)

        roi_order = copy(rois)
        random.shuffle(roi_order)
        trial['roiOrder'] = roi_order

        if use_same_img:
            gen_idx = orig_idx
        else:
            gen_idx = random.sample([i for i in range(n_per_cat) if i != orig_idx], 1)[0]

        trial['gen'] = []
        for roi in roi_order:
            sample_idx = random.randint(0, n_samples - 1)
            gen_name = '{}_{:03d}_{}_{}.jpg'.format(cat, gen_idx, sample_idx, roi)
            shutil.copy(generated_dir[cond] + '{}/{}_{:03d}_{}.jpg'.format(roi, cat, gen_idx, sample_idx),
                        img_dir + gen_name)
            trial['gen'].append(gen_name)
            trial_data['images'].append(gen_name)

        trial_data['trainTrials'].append(trial)

for cat, cond in exp_cats:
    for orig_idx in range(n_per_exp):
        trial = {'condition': cond}

        orig_name = '{}_{:03d}.jpg'.format(cat, orig_idx)
        shutil.copy(orig_dir[cond] + orig_name, img_dir + orig_name)
        trial['orig'] = orig_name
        trial_data['images'].append(orig_name)

        roi_order = copy(rois)
        random.shuffle(roi_order)
        trial['roiOrder'] = roi_order

        if use_same_img:
            gen_idx = orig_idx
        else:
            gen_idx = random.sample([i for i in range(n_per_cat) if i != orig_idx], 1)[0]

        trial['gen'] = []
        for roi in roi_order:
            sample_idx = random.randint(0, n_samples - 1)
            gen_name = '{}_{:03d}_{}_{}.jpg'.format(cat, gen_idx, sample_idx, roi)
            shutil.copy(generated_dir[cond] + '{}/{}_{:03d}_{}.jpg'.format(roi, cat, gen_idx, sample_idx),
                        img_dir + gen_name)
            trial['gen'].append(gen_name)
            trial_data['images'].append(gen_name)

        trial_data['experimentTrials'].append(trial)

for _ in range(n_catch):
    trial = {'condition': 'catch'}
    cat, cond = random.sample(train_cats, 1)[0]
    orig_idx = random.sample(range(n_per_cat), 1)[0]
    orig_name = '{}_{:03d}.jpg'.format(cat, orig_idx)
    if not os.path.exists(img_dir + orig_name):
        shutil.copy(orig_dir[cond] + orig_name, img_dir + orig_name)
    trial['orig'] = orig_name
    trial_data['images'].append(orig_name)

    answer_order = ['correct'] + ['incorrect' for _ in range(len(rois) - 1)]
    random.shuffle(answer_order)
    trial['roiOrder'] = answer_order

    trial['gen'] = []
    for answer in answer_order:
        if answer == 'correct':
            trial['gen'].append(orig_name)
        else:
            roi = random.sample(rois, 1)[0]
            sample_idx = random.randint(0, n_samples - 1)
            gen_name = '{}_{:03d}_{}_{}.jpg'.format(cat, orig_idx, sample_idx, roi)
            if not os.path.exists(img_dir + gen_name):
                shutil.copy(generated_dir[cond] + '{}/{}_{:03d}_{}.jpg'.format(roi, cat, orig_idx, sample_idx),
                            img_dir + gen_name)
            trial['gen'].append(gen_name)
            trial_data['images'].append(gen_name)

    trial_data['experimentTrials'].append(trial)

trial_data['images'] = list(set(trial_data['images']))
with open(save_dir + 'trialData.json', 'w') as f:
    f.write(json.dumps(trial_data, indent=2))
