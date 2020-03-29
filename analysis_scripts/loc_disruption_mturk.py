import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style='whitegrid')


def transform_data(data, keep_catch=False):
    responses, types, reaction_times, images = data['Answer.trialResponses'], data['Answer.types'], \
                                               data['Answer.reactionTimes'], data['Answer.displayedImages']

    def split(x):
        return x.split(',')
    def split_num(x):
        x = split(x)
        return [float(a) for a in x]
    responses, types, reaction_times, images = responses.apply(split_num), types.apply(split), \
                                               reaction_times.apply(split_num), images.apply(split)

    transformed = {'Type': [], 'Subject': [], 'Response': [], 'Reaction Time': [], 'Image': []}
    for s, (st, sr, srt, simg) in enumerate(zip(types, responses, reaction_times, images)):
        for t, r, rt, img in zip(st, sr, srt, simg):
            if not keep_catch and t == 'catch':
                continue
            transformed['Type'].append(t)
            transformed['Subject'].append(s)
            transformed['Response'].append(r)
            transformed['Reaction Time'].append(rt)
            transformed['Image'].append(img.replace('.JPEG', ''))

    transformed = pd.DataFrame(transformed)
    return transformed


def plot_results(data, y='Response'):
    plt.close()
    ax = sns.barplot(x='Type', y=y, data=data,
                     order=['original', 'disrupted', 'random'])
    plt.show()


def plot_results_by_image(data, y='Response'):
    plt.close()
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(x='Image', y=y, hue='Type', data=data,
                     order=sorted(set(data['Image'])), hue_order=['original', 'disrupted', 'random'], errwidth=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()


data = pd.read_csv('/home/eric/Documents/experiments/adversarial_tms/mturk_results/loc_disruption_results.csv')
data = transform_data(data)
# plot_results(data, 'Response')
# plot_results(data, 'Reaction Time')
# plot_results_by_image(data, 'Response')
# plot_results_by_image(data, 'Reaction Time')
