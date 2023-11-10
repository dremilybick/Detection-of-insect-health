import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os

field = 'corn1'

# Load data
snippets = pd.read_csv(f'data/processed/{field}_snippets.csv')
df = pd.read_csv(f'data/processed/{field}_filtered.csv')
sample_rate = 22050

# How close do events need to be before we can categorise them as the same event?
merge_seconds = 0.01
merge_frames = int(sample_rate * merge_seconds)

# Set a minimum event length
min_length = 0


# Function to merge intervals
def merge_intervals(intervals, distance):
    merged = []
    current_start, current_stop = intervals[0]

    for start, stop in intervals[1:]:
        if start - current_stop <= distance:
            current_stop = stop
        else:
            merged.append((current_start, current_stop))
            current_start, current_stop = start, stop

    merged.append((current_start, current_stop))
    return merged


# Merge intervals
merged_intervals = merge_intervals(snippets[['ix1', 'ix2']].values.tolist(), merge_frames)

# Convert merged intervals back to DataFrame
merged_df = pd.DataFrame(merged_intervals, columns=['ix1', 'ix2'])

# Calculate length of each snippet
merged_df['frames'] = (merged_df['ix2'] - merged_df['ix1']) + 1
merged_df['length'] = merged_df['frames'] / sample_rate

# Plot a histogram
plt.hist(merged_df['length'], bins=50)
plt.xlabel('Length (seconds)')
plt.ylabel('N events')

# Make folders
if not os.path.exists(f'results/plots/{field}/'):
    os.makedirs(f'results/plots/{field}/')
if not os.path.exists(f'results/data/{field}/'):
    os.makedirs(f'results/data/{field}/')

# Create events for plotting and analysis
event_num = 0
snippets = []
for i, row in merged_df.iterrows():
    if row['length'] < min_length:
        continue
    event_num += 1
    snip = df.iloc[int(row['ix1']) - 100:int(row['ix2']) + 200]
    length = (int(row['ix2']) - int(row['ix1'])) / sample_rate
    start_time = snip.index[0] / sample_rate
    snip = snip.reset_index(drop=True)
    snip['frames'] = snip.index
    snippets.append([list(snip['treat'])])
    plt.figure()
    plt.plot(snip['frames'], snip['treat'])
    plt.xlabel('Frames')
    plt.title(f'Start time: {str(datetime.timedelta(seconds=round(start_time)))}, length: {round(length, 5)}s')
    plt.savefig(f'results/plots/{field}/{event_num}.png')
    plt.close('all')
snippets = pd.DataFrame(snippets, columns=['event'])
snippets['length'] = snippets['event'].str.len() - 300
snippets['intensity'] = snippets['event'].apply(max)
snippets.to_csv(f'results/data/{field}/events.csv', index=False)


# Intensity/length plots
def damped_oscillator(x, m, c):
    return (-2*m/c)*(np.log(0.0001)-np.log(x))
def estimate_damping_coefficient(t, x0):
    tau = t / np.log(x0 / (x0 - 0.001))
    c_estimate = 2 * x0 / (tau * t)
    return c_estimate

# Estimate the damping coefficient by taking the median
cs = []
for i, row in snippets.iterrows():
    c_est = estimate_damping_coefficient(row['length'], row['intensity'])
    cs.append(c_est)
cest =np.median(cs)

# Generate the damped oscillator trend line - values 0.0013 and -80 found empirically
x_vals = np.arange(min(snippets['intensity']),max(snippets['intensity']),max(snippets['intensity'])/100)
y = [damped_oscillator(x, 0.0013, cest)-80 for x in x_vals]
plt.plot(x_vals, y)

# Plot intensity/length of data
plt.scatter(snippets['intensity'], snippets['length'])
plt.xlabel('max intensity')
plt.ylabel('length')
plt.tight_layout()