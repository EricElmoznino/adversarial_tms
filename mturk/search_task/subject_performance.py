import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sns.set(style='whitegrid')

result_file = 'may21.csv'


def load_data(max_rt=None, filter_correct=False):
    data_raw = pd.read_csv('results/' + result_file)
    data_raw['subjects'] = np.arange(len(data_raw))

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

    if filter_correct:
        data = data[data['corrects'] == 1]
    if max_rt is not None:
        data = data[data['reactionTimes'] < max_rt]

    return data


data = load_data()
print(len(data))

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
