import os
import shutil
import json
import random
from PIL import Image

random.seed(27)

orig_dir = '/home/eric/Documents/datasets/adversarial_tms/mturkscenes/'
generated_dir = '/home/eric/Documents/experiments/adversarial_tms/gan_manipulations/bold5000/mturkscenes/conv3/ppa/'
save_dir = '/home/eric/Desktop/mturk_images/'

n_per_cat = 6
n_samples = 10
n_per_trial = 6

train_cats = ['mountain', 'boathouse', 'store']
exp_cats = ['beach', 'street', 'kitchen', 'forest', 'building', 'office']
n_per_train = 2
n_per_exp = 3

n_catches = 0

shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)

trial_data = {'trainTrials': [], 'experimentTrials': []}

for c in train_cats:
    target_nums = random.sample(range(n_per_cat), 2 * n_per_train)

    # Category-consistent
    for t in target_nums[:n_per_train]:
        foil_nums = random.sample([f for f in range(n_per_cat) if f != t], n_per_trial - 1)
        foil_samples = [random.randint(0, n_samples - 1) for _ in range(n_per_trial)]
        order = [t] + foil_nums
        random.shuffle(order)
        trial_name = '{}{:03d}_{}'.format(c, t, c)
        os.mkdir(save_dir + trial_name)
        for i, num in enumerate(order):
            if num == t:
                trial_data['trainTrials'].append({'image': trial_name, 'true': c, 'foil': c, 'answer': i})
                shutil.copy(orig_dir + '{}_{:03d}.jpg'.format(c, t),
                            save_dir + trial_name + '/{:03d}_target.jpg'.format(i))
            else:
                shutil.copy(generated_dir + '{}_{:03d}_{}.png'.format(c, num, foil_samples[i]),
                            save_dir + trial_name + '/{:03d}.png'.format(i))

    # Category-inconsistent
    foil_cats = random.sample([cat for cat in train_cats if cat != c], n_per_train)
    for t, fc in zip(target_nums[n_per_train:], foil_cats):
        foil_nums = random.sample([f for f in range(n_per_cat) if f != t], n_per_trial - 1)
        foil_samples = [random.randint(0, n_samples - 1) for _ in range(n_per_trial)]
        order = [t] + foil_nums
        random.shuffle(order)
        trial_name = '{}{:03d}_{}'.format(c, t, fc)
        os.mkdir(save_dir + trial_name)
        for i, num in enumerate(order):
            if num == t:
                trial_data['trainTrials'].append({'image': trial_name, 'true': c, 'foil': fc, 'answer': i})
                shutil.copy(orig_dir + '{}_{:03d}.jpg'.format(c, t),
                            save_dir + trial_name + '/{:03d}_target.jpg'.format(i))
            else:
                shutil.copy(generated_dir + '{}_{:03d}_{}.png'.format(fc, num, foil_samples[i]),
                            save_dir + trial_name + '/{:03d}.png'.format(i))

for c in exp_cats:
    target_nums = random.sample(range(n_per_cat), 2 * n_per_exp)

    # Category-consistent
    for t in target_nums[:n_per_exp]:
        foil_nums = random.sample([f for f in range(n_per_cat) if f != t], n_per_trial - 1)
        foil_samples = [random.randint(0, n_samples - 1) for _ in range(n_per_trial)]
        order = [t] + foil_nums
        random.shuffle(order)
        trial_name = '{}{:03d}_{}'.format(c, t, c)
        os.mkdir(save_dir + trial_name)
        for i, num in enumerate(order):
            if num == t:
                trial_data['experimentTrials'].append({'image': trial_name, 'true': c, 'foil': c, 'answer': i,
                                                       'isCatch': False})
                shutil.copy(orig_dir + '{}_{:03d}.jpg'.format(c, t),
                            save_dir + trial_name + '/{:03d}_target.jpg'.format(i))
            else:
                shutil.copy(generated_dir + '{}_{:03d}_{}.png'.format(c, num, foil_samples[i]),
                            save_dir + trial_name + '/{:03d}.png'.format(i))

    # Category-inconsistent
    foil_cats = random.sample([cat for cat in exp_cats if cat != c], n_per_exp)
    for t, fc in zip(target_nums[n_per_exp:], foil_cats):
        foil_nums = random.sample([f for f in range(n_per_cat) if f != t], n_per_trial - 1)
        foil_samples = [random.randint(0, n_samples - 1) for _ in range(n_per_trial)]
        order = [t] + foil_nums
        random.shuffle(order)
        trial_name = '{}{:03d}_{}'.format(c, t, fc)
        os.mkdir(save_dir + trial_name)
        for i, num in enumerate(order):
            if num == t:
                trial_data['experimentTrials'].append({'image': trial_name, 'true': c, 'foil': fc, 'answer': i,
                                                       'isCatch': False})
                shutil.copy(orig_dir + '{}_{:03d}.jpg'.format(c, t),
                            save_dir + trial_name + '/{:03d}_target.jpg'.format(i))
            else:
                shutil.copy(generated_dir + '{}_{:03d}_{}.png'.format(fc, num, foil_samples[i]),
                            save_dir + trial_name + '/{:03d}.png'.format(i))

catch_true = Image.new('RGB', (256, 256), color=(0, 200, 0))
catch_foil = Image.new('RGB', (256, 256), color=(200, 0, 0))
for catch_num in range(n_catches):
    trial_name = 'catch{:03d}'.format(catch_num)
    os.mkdir(save_dir + trial_name)
    true_num = random.randint(0, n_per_trial - 1)
    for i in range(n_per_trial):
        if i == true_num:
            trial_data['experimentTrials'].append({'image': trial_name, 'true': 'catch', 'foil': 'catch', 'answer': i,
                                                   'isCatch': True})
            catch_true.save(save_dir + trial_name + '/{:03d}_target.jpg'.format(i))
        else:
            catch_foil.save(save_dir + trial_name + '/{:03d}.jpg'.format(i))

with open(save_dir + 'trialData.json', 'w') as f:
    f.write(json.dumps(trial_data, indent=2))
