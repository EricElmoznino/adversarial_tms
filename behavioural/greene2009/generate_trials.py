import os
import shutil
import json
import random
import pandas as pd

random.seed(27)

attribute = 'concealment'
orig_dir = '/home/eric/datasets/adversarial_tms/greene2009'
generated_dir = '/home/eric/experiments/adversarial_tms/gan_manipulations/bold5000/greene2009/conv3'
rois = ['loc', 'ppa']
save_dir = '/home/eric/Desktop/' + attribute

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)
for condition in rois + ['catch']:
    os.mkdir(os.path.join(save_dir, condition))

train_per_roi = 4
test_per_roi = 30
n_catch = 6

trials = {}

# Load the images and split them by their corresponding attribute
data = pd.read_csv(os.path.join(orig_dir, 'key.csv'))
neg_images = data[attribute.capitalize()][data[attribute.capitalize() + '.Answer'] == 0].tolist()
pos_images = data[attribute.capitalize()][data[attribute.capitalize() + '.Answer'] == 1].tolist()
assert test_per_roi + train_per_roi <= min(len(neg_images), len(pos_images))
random.shuffle(neg_images)
random.shuffle(pos_images)
neg_train_images = neg_images[:train_per_roi]
pos_train_images = pos_images[:train_per_roi]
neg_test_images = neg_images[train_per_roi:train_per_roi + test_per_roi]
pos_test_images = pos_images[train_per_roi:train_per_roi + test_per_roi]
neg_catch_images = neg_images[-n_catch:]
pos_catch_images = pos_images[-n_catch:]

# Copy over train trials
for roi in rois:
    for pos, neg in zip(pos_train_images, neg_train_images):
        shutil.copyfile(os.path.join(generated_dir, roi, pos.replace('.jpg', '_0.jpg')),
                        os.path.join(save_dir, roi, pos))
        shutil.copyfile(os.path.join(generated_dir, roi, neg.replace('.jpg', '_0.jpg')),
                        os.path.join(save_dir, roi, neg))
trials['train'] = {
    'positives': pos_train_images,
    'negatives': neg_train_images
}

# Copy over test trials
for roi in rois:
    for pos, neg in zip(pos_test_images, neg_test_images):
        shutil.copyfile(os.path.join(generated_dir, roi, pos.replace('.jpg', '_0.jpg')),
                        os.path.join(save_dir, roi, pos))
        shutil.copyfile(os.path.join(generated_dir, roi, neg.replace('.jpg', '_0.jpg')),
                        os.path.join(save_dir, roi, neg))
trials['test'] = {
    'positives': pos_test_images,
    'negatives': neg_test_images
}

# Copy over catch trials
for pos, neg in zip(pos_catch_images, neg_catch_images):
    shutil.copyfile(os.path.join(orig_dir, 'all', pos),
                    os.path.join(save_dir, 'catch', pos))
    shutil.copyfile(os.path.join(orig_dir, 'all', neg),
                    os.path.join(save_dir, 'catch', neg))
trials['catch'] = {
    'positives': pos_catch_images,
    'negatives': neg_catch_images
}

with open(os.path.join(save_dir, 'trials.json'), 'w') as f:
    f.write(json.dumps(trials, indent=2))
