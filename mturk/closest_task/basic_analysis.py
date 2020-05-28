import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid')

result_file = 'may22.csv'
rt_range = None


def load_data(keep_catches=False):
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

    if not keep_catches:
        data = data[data['isCatch'] == 1]

    return data


data = load_data()

# Plot frequencies of different ROIs
plt.close()
g = sns.catplot(kind='count', x='response', col='Subject', col_wrap=2, data=data)
for ax in g.axes:
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.subplots_adjust(hspace=0.2, wspace=0.1)
plt.show()
