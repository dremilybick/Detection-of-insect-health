import pandas as pd
import numpy as np
def load_data(file):
    df = pd.read_csv(file)
    signals = df['snippets'].apply(eval).apply(np.array)
    thresholds = df['thres_signal']
    origin = df['file_name']
    return signals, thresholds, origin