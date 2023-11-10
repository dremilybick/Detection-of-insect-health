import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa

# Open control signal
control = 'data/examples/DataN/corncontrol1.wav'
control_signal, control_sample_rate = librosa.load(control)

# Open treatment signal
treatment = 'data/examples/DataN/corntreatment1.wav'
treatment_signal, treat_sample_rate = librosa.load(treatment)

# Remove 60Hz noise and first 6 harmonics using notch filter
exclusion_freqs = [60, 120, 180, 240, 300, 360, 420]
fs = control_sample_rate
Q = 30.0
out_control = control_signal.copy()
out_treat = treatment_signal.copy()
for f0 in exclusion_freqs:
    b, a = signal.iirnotch(f0, Q, fs)
    out_control = signal.filtfilt(b, a, out_control)
    out_treat = signal.filtfilt(b, a, out_treat)

# Find standard deviation of each signal
thres_treat = np.std(out_treat)
thres_control = np.std(out_control)

# Get absolute value of signal
out_treat = np.abs(out_treat)
out_control = np.abs(out_control)

# Identify regions of interest where signal exceeds 20x standard deviation
multiplier = 20
df = pd.DataFrame({'treat': out_treat, 'control': out_control})
df.loc[df['control'] > thres_control * multiplier, 'control_thres'] = df['control']
df.loc[df['treat'] > thres_treat * multiplier, 'treat_thres'] = df['treat']


# Look at a sample of this to get an idea of how much of the signal this has selected (show >threshold regions in red)
sdf = df[66000000:68000000]
fig, axs = plt.subplots(2, 1)
axs[0].plot(sdf['control'], 'k')
axs[0].plot(sdf['control_thres'], 'r')
axs[0].set_title('Control')
axs[1].plot(sdf['treat'], 'k')
axs[1].plot(sdf['treat_thres'], 'r')
axs[1].set_title('Treatment')

# Extract continuous 'snippets' which are above the threshold
control_snippets = []
treat_snippets = []
start_treat = None
start_control = None
treat_max = 0
control_max = 0
for i, row in tqdm(df.iterrows(), total=len(df)):
    if not np.isnan(row['treat_thres']):
        if start_treat is None:
            start_treat = i
            treat_max = row['treat_thres']
        elif row['treat_thres'] > treat_max:
            treat_max = row['treat_thres']
    elif start_treat is not None:
        treat_snippets.append([start_treat, i - 1, treat_max])
        start_treat = None
        treat_max = 0
    if not np.isnan(row['control_thres']):
        if start_control is None:
            start_control = i
            control_max = row['control_thres']
        elif row['control_thres'] > treat_max:
            treat_max = row['control_thres']
    elif start_control is not None:
        control_snippets.append([start_control, i - 1, control_max])
        start_control = None
        control_max = 0

# Collect them together in a handy dataframe
control_snippets = pd.DataFrame(control_snippets, columns=['ix1', 'ix2', 'height'])
treat_snippets = pd.DataFrame(treat_snippets, columns=['ix1', 'ix2', 'height'])
control_snippets['condition'] = 'control'
treat_snippets['condition'] = 'treatment'
snippets = pd.concat([control_snippets, treat_snippets])

snippets['length'] = snippets['ix2'] - snippets['ix1'] + 1

# Print a summary
for condition in snippets['condition'].unique():
    print(condition)
    print(f"    n :          {len(snippets[snippets['condition'] == condition])} \n"
          f"    avg length : {snippets[snippets['condition'] == condition]['length'].mean()} \n"
          f"    max length : {snippets[snippets['condition'] == condition]['length'].max()} \n"
          f"    avg height : {snippets[snippets['condition'] == condition]['height'].mean()} \n"
          f"    max height : {snippets[snippets['condition'] == condition]['height'].max()}")

# Save them for later
snippets.to_csv('data/processed/corn2_snippets.csv', index=False)
df.to_csv('data/processed/corn2_filtered.csv', index=False)
