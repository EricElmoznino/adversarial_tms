import os
import json
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid')

attribute = 'concealment'
aggregated = False


def load_data(catch_thresh=None):
    files = os.listdir('results/{}'.format(attribute))
    files = ['results/{}/{}'.format(attribute, f) for f in files if '.json' in f]
    data = []
    for file in files:
        with open(file, 'r') as f:
            data.append(json.loads(f.read()))

    data = pd.DataFrame(data)
    data['Subject'] = np.arange(1, len(data) + 1)

    def expand_trials(df, trials_per_subj):
        return pd.DataFrame({field: np.repeat(df[field].values, trials_per_subj)
                             for field in df.columns if not isinstance(df[field].values[0], list)}
                            ).assign(**{field: np.concatenate(df[field].values)
                                        for field in df.columns if isinstance(df[field].values[0], list)})
    trials_per_subj = len(data['Response'].values[0])
    data = expand_trials(data, trials_per_subj)

    data['Consistent'] = data['Response'] == 'pos'

    if catch_thresh is not None:
        n_subj = len(files)
        n_catches = len(data[data['Condition'] == 'catch']) / n_subj
        catch_thresh = math.ceil(n_catches * catch_thresh)
        data = data.groupby(['Subject']).filter(lambda x: x['Response'].eq('pos').sum() >= catch_thresh)
        data = data[data['Condition'] != 'catch']

    return data


data = load_data(catch_thresh=1.0)

# Plot frequencies of different ROIs
plt.close()
if aggregated:
    g = sns.catplot(kind='bar', x='Condition', y='Consistent', data=data)
else:
    g = sns.catplot(kind='bar', x='Condition', y='Consistent',
                    col='Subject', col_wrap=2, data=data)
plt.show()
