import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid')

result_file = 'may28.csv'
aggregrated = False


def load_data(catch_thresh=0.8):
    data_raw = pd.read_csv('results/' + result_file)
    data_raw['Subject'] = np.arange(1, len(data_raw) + 1)
    data_raw['response'] = data_raw['response'].apply(lambda x: x.upper())

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
    trials_per_subj = len(data['response'].values[0])
    data = expand_trials(data, trials_per_subj)

    if catch_thresh is not None:
        n_subj = len(data_raw)
        n_catches = len(data[data['condition'] == 'catch']) / n_subj
        catch_thresh = math.ceil(n_catches * catch_thresh)
        data = data.groupby(['Subject']).filter(lambda x: x['response'].eq('CORRECT').sum() >= catch_thresh)
        data = data[data['condition'] != 'catch']

    return data


data = load_data()

# Plot frequencies of different ROIs
plt.close()
if aggregrated:
    sns.catplot(kind='count', x='response', hue='condition', data=data)
else:
    g = sns.catplot(kind='count', x='response', hue='condition', col='Subject', col_wrap=2, data=data)
    for ax in g.axes:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
plt.show()
