import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sns.set(style='whitegrid')

result_file = 'may21.csv'
rt_range = (0, 9000)


def load_data():
    data_raw = pd.read_csv('results/' + result_file)
    data_raw['subjects'] = np.arange(1, len(data_raw) + 1)

    def split_trials(row):
        def split_inner(cell):
            if isinstance(cell, str) and ',' in cell:
                cell = cell.split(',')
                if cell[0].isnumeric():
                    cell = [float(a) for a in cell]
            return cell
        return row.apply(split_inner)
    data = data_raw.apply(split_trials)

    def expand_trials(df, trials_per_subj):
        return pd.DataFrame({field: np.repeat(df[field].values, trials_per_subj)
                             for field in df.columns if not isinstance(df[field].values[0], list)}
                            ).assign(**{field: np.concatenate(df[field].values)
                                        for field in df.columns if isinstance(df[field].values[0], list)})
    trials_per_subj = len(data['corrects'].values[0])
    data = expand_trials(data, trials_per_subj)

    return data


def filter_data(data, is_correct=False, rt_range=None):
    if is_correct:
        data = data[data['corrects'] == 1]
    if rt_range is not None:
        data = data[(rt_range[0] < data['reactionTimes']) & (data['reactionTimes'] < rt_range[1])]
    return data


data = load_data()

# Plot raw subject performance to see if data looks good and identify outliers
fig, ax = plt.subplots(1, 2)
sns.scatterplot(x='subjects', y='reactionTimes', data=data, ax=ax[0])
sns.barplot(x='subjects', y='corrects', data=data, ax=ax[1])
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0].set_xlabel('Subject')
ax[0].set_ylabel('Reaction Time')
ax[1].set_xlabel('Subject')
ax[1].set_ylabel('Accuracy')
fig.tight_layout()
plt.show()

# Plot reaction times by ROI
plt.close()
ax = sns.violinplot(x='subjects', y='reactionTimes', hue='rois',
                    data=filter_data(data, is_correct=True, rt_range=rt_range))
ax.set_xlabel('Subject')
ax.set_ylabel('Reaction Time')
plt.show()

# Plot accuracy by ROI
plt.close()
ax = sns.barplot(x='subjects', y='corrects', hue='rois',
                 data=filter_data(data, rt_range=rt_range))
ax.set_xlabel('Subject')
ax.set_ylabel('Accuracy')
plt.show()
