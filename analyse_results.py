import pandas as pd

df = pd.read_csv('resultanalysis.csv')

result = df[['file_name','insect_name','type','thres_signal','num_subsignals']]

for col in ['merged_intervals','snippets','snippets_length','length','max']:
    subset = df[['file_name', col]]

    if any(subset[col].str.contains('array')):
        subset[col] = subset[col].str.replace('array(','')
        subset[col] = subset[col].str.replace(')', '')
    subset[col] = subset[col].apply(lambda x: eval(x))

    # Duplicate rows based on the elements in the lists in column 'c'
    subset = subset.explode(col)

    if len(subset)!=5938:
        raise Exception("Oh no")

    subset = subset.reset_index(drop=True).reset_index()

    if 'index' in result:
        result = pd.merge(result, subset, on=['file_name','index'])
    else:
        result = pd.merge(result, subset, on=['file_name'])

result.to_csv('long_format_results.csv')

#%%
import numpy as np

result = pd.read_csv('long_format_results.csv')
toolong = result[result['snippets'].str.contains('Ellipsis').fillna(False)]
okay =  result[~result['snippets'].str.contains('Ellipsis').fillna(True)]
okay['snippets'] = okay['snippets'].apply(lambda x: eval(x))

# Any snippets > 700 values (0.03184 seconds) are lost due to wide storage

def precise_floor(x, dp):
    return np.true_divide(np.floor(x*10**dp),10**dp)


okay['strict_length'] = None
for i, row in okay.iterrows():
    snippet = abs(np.array(row['snippets']))
    threshold = row['thres_signal']
    intensity = precise_floor(row['max'], 5)

    start = np.argmax(snippet>intensity)
    stop = np.flatnonzero(snippet>threshold)[-1]

    okay.at[i, 'strict_length'] = stop-start


okay.to_csv('strict_length_fixed.csv')

result = okay

#%%

import matplotlib.pyplot as plt
import numpy as np

#result = result[result['length']>3]

plt.hist(okay[okay['strict_length']<100]['strict_length'], bins = 50)
plt.show()

for insect_name in result['insect_name'].unique():
    plt.figure()
    subset = result[result['insect_name']==insect_name]
    plt.scatter(subset[subset['type']=='Treatment']['max'],subset[subset['type']=='Treatment']['strict_length'], label='Treatment')
    plt.scatter(subset[subset['type']=='Control']['max'],subset[subset['type']=='Control']['strict_length'], label='Control')
    plt.xlabel('Intensity')
    plt.ylabel('Length')
    plt.legend()
    plt.title(insect_name)
    plt.tight_layout()


min_length, max_length = (250,255)
min_intensity, max_intensity = (0.094,0.104)

min_length, max_length = (261,263)
min_intensity, max_intensity = (0.0122,0.0127)

ex = subset[(subset['strict_length']>min_length)&
            (subset['strict_length']<max_length)&
            (subset['max']>min_intensity)&
            (subset['max']<max_intensity)]
ex_cont = ex[ex['type']=='Control'].iloc[0]
ex_treat = ex[ex['type']=='Treatment'].iloc[0]

fig, axs = plt.subplots(2,1)
axs[0].plot(np.abs(ex_cont['snippets']))
axs[0].set_title('Control')
axs[1].plot(np.abs(ex_treat['snippets']))
axs[1].set_title('Treatment')