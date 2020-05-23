import os
import shutil
from copy import copy
import json
import random

random.seed(27)

orig_dir = '/home/eric/Documents/datasets/adversarial_tms/mturkscenes/'
generated_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/mturkscenes/conv3/'
save_dir = 'web/assets/'

use_same_img = False
n_catch = 6
n_per_cat = 9
n_samples = 10
rois = ['ppa', 'loc', 'evc', 'random']

train_cats = ['mountain', 'boathouse', 'store']
exp_cats = ['beach', 'street', 'kitchen', 'forest', 'building', 'office']
n_per_train = 2
n_per_exp = 9

img_dir = save_dir + 'images/'
trial_data_path = save_dir + 'trialData.json'
shutil.rmtree(img_dir, ignore_errors=True)
os.mkdir(img_dir)
shutil.rmtree(trial_data_path, ignore_errors=True)

trial_data = {'images': [], 'trainTrials': [], 'experimentTrials': []}

for cat in train_cats:
    for orig_idx in range(n_per_train):
        trial = {'isCatch': False}

        orig_name = '{}_{:03d}.jpg'.format(cat, orig_idx)
        shutil.copy(orig_dir + orig_name, img_dir + orig_name)
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
            gen_name = '{}_{:03d}_{}_{}.png'.format(cat, gen_idx, sample_idx, roi)
            shutil.copy(generated_dir + '{}/{}_{:03d}_{}.png'.format(roi, cat, gen_idx, sample_idx),
                        img_dir + gen_name)
            trial['gen'].append(gen_name)
            trial_data['images'].append(gen_name)

        trial_data['trainTrials'].append(trial)

for cat in exp_cats:
    for orig_idx in range(n_per_exp):
        trial = {'isCatch': False}

        orig_name = '{}_{:03d}.jpg'.format(cat, orig_idx)
        shutil.copy(orig_dir + orig_name, img_dir + orig_name)
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
            gen_name = '{}_{:03d}_{}_{}.png'.format(cat, gen_idx, sample_idx, roi)
            shutil.copy(generated_dir + '{}/{}_{:03d}_{}.png'.format(roi, cat, gen_idx, sample_idx),
                        img_dir + gen_name)
            trial['gen'].append(gen_name)
            trial_data['images'].append(gen_name)

        trial_data['experimentTrials'].append(trial)

for _ in range(n_catch):
    trial = {'isCatch': True}

    cat = random.sample(train_cats, 1)[0]
    orig_idx = random.sample(range(n_per_cat), 1)[0]
    orig_name = '{}_{:03d}.jpg'.format(cat, orig_idx)
    if not os.path.exists(img_dir + orig_name):
        shutil.copy(orig_dir + orig_name, img_dir + orig_name)
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
            gen_name = '{}_{:03d}_{}_{}.png'.format(cat, orig_idx, sample_idx, roi)
            if not os.path.exists(img_dir + gen_name):
                shutil.copy(generated_dir + '{}/{}_{:03d}_{}.png'.format(roi, cat, orig_idx, sample_idx),
                            img_dir + gen_name)
            trial['gen'].append(gen_name)
            trial_data['images'].append(gen_name)

    trial_data['experimentTrials'].append(trial)

trial_data['images'] = list(set(trial_data['images']))
with open(save_dir + 'trialData.json', 'w') as f:
    f.write(json.dumps(trial_data, indent=2))
