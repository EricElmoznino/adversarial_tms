import os
import shutil
import json
import random

random.seed(27)

orig_dir = '/home/eric/Documents/datasets/adversarial_tms/mturkscenes/'
generated_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/mturkscenes/conv3/'
save_dir = '/home/eric/Desktop/mturk_images/'

n_per_cat = 9
n_samples = 10
n_per_trial = 6
rois = ['ppa', 'loc', 'evc']

train_cats = ['mountain', 'boathouse', 'store']
exp_cats = ['beach', 'street', 'kitchen', 'forest', 'building', 'office']
n_per_train = 2
n_per_exp = 9

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)

trial_data = {'trainTrials': [[] for _ in range(len(rois))],
              'experimentTrials': [[] for _ in range(len(rois))]}

for c in train_cats:
    target_nums = random.sample(range(n_per_cat), n_per_train)
    for roi_offset, t in enumerate(target_nums):
        roi_order = [rois[(i + roi_offset) % len(rois)] for i in range(len(rois))]
        hit_sets = [i for i in range(len(rois))]
        random.shuffle(hit_sets)
        for hset, roi in enumerate(roi_order):
            foil_nums = random.sample([f for f in range(n_per_cat) if f != t], n_per_trial - 1)
            foil_samples = [random.randint(0, n_samples - 1) for _ in range(n_per_trial)]
            order = [t] + foil_nums
            random.shuffle(order)
            trial_name = '{}{:03d}_{}'.format(c, t, roi)
            os.mkdir(save_dir + trial_name)
            for i, num in enumerate(order):
                if num == t:
                    trial_data['trainTrials'][hset].append({'image': trial_name, 'roi': roi, 'answer': i})
                    shutil.copy(orig_dir + '{}_{:03d}.jpg'.format(c, t),
                                save_dir + trial_name + '/{:03d}_target.jpg'.format(i))
                else:
                    shutil.copy(generated_dir + '/' + roi + '/{}_{:03d}_{}.jpg'.format(c, num, foil_samples[i]),
                                save_dir + trial_name + '/{:03d}.jpg'.format(i))

for c in exp_cats:
    target_nums = random.sample(range(n_per_cat), n_per_exp)
    for roi_offset, t in enumerate(target_nums):
        roi_order = [rois[(i + roi_offset) % len(rois)] for i in range(len(rois))]
        hit_sets = [i for i in range(len(rois))]
        random.shuffle(hit_sets)
        for hset, roi in enumerate(roi_order):
            foil_nums = random.sample([f for f in range(n_per_cat) if f != t], n_per_trial - 1)
            foil_samples = [random.randint(0, n_samples - 1) for _ in range(n_per_trial)]
            order = [t] + foil_nums
            random.shuffle(order)
            trial_name = '{}{:03d}_{}'.format(c, t, roi)
            os.mkdir(save_dir + trial_name)
            for i, num in enumerate(order):
                if num == t:
                    trial_data['experimentTrials'][hset].append({'image': trial_name, 'roi': roi, 'answer': i})
                    shutil.copy(orig_dir + '{}_{:03d}.jpg'.format(c, t),
                                save_dir + trial_name + '/{:03d}_target.jpg'.format(i))
                else:
                    shutil.copy(generated_dir + '/' + roi + '/{}_{:03d}_{}.jpg'.format(c, num, foil_samples[i]),
                                save_dir + trial_name + '/{:03d}.jpg'.format(i))

with open(save_dir + 'trialData.json', 'w') as f:
    f.write(json.dumps(trial_data, indent=2))
