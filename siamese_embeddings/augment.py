import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

def augment(snippet, noise, shuffle_ratio):
    augmented = snippet.copy()
    augmented = noiser(augmented, noise)
    scale = np.random.random() + 0.5
    augmented = rescale(augmented, scale, noise)
    augmented = shuffle(augmented, shuffle_ratio)

    return augmented

def rescale(array, scale, noise, threshold=20):
    out_array = [x*scale if x>noise*threshold else x for x in array]
    indices = np.linspace(0, len(array) - 1, int(len(array) * scale))
    out_array = np.interp(indices, np.arange(len(array)), out_array)

    return out_array

def noiser(array, noise, average=False):
    noise_signal = np.random.normal(0, noise, len(array))
    added = array +noise_signal
    if average:
        return np.mean([added, array], axis=0)
    return added

def shuffle(array, percentage):
    swap_ix = np.random.choice(len(array)-1, int(len(array)*percentage), replace=False)
    out_array = array.copy()
    for ix in swap_ix:
        out_array[ix], out_array[ix+1] = out_array[ix+1], out_array[ix]
    return out_array

def crop_and_pad(array, noise, left_length=100, right_length=300):
    in_array = array.copy()
    start_max_ix = np.argmax(in_array)
    pad_left = left_length - start_max_ix
    pad_right = right_length - (len(array)-start_max_ix)
    if pad_left>0:
        noise_signal = np.minimum(np.abs(np.random.normal(0, noise, pad_left)),in_array[start_max_ix]-0.0000000000001)
        in_array = np.concatenate([noise_signal, in_array])
    if pad_right>0:
        noise_signal = np.minimum(np.abs(np.random.normal(0, noise, pad_right)),in_array[start_max_ix]-0.0000000000001)
        in_array = np.concatenate([in_array, noise_signal])
    max_ix = np.argmax(in_array)
    in_array = in_array[max_ix-left_length:max_ix+right_length]

    if len(in_array)!=left_length+right_length:
        raise Exception

    return in_array


def data_generator(data_type, noise=0.1):
    array = []
    length = np.random.randint(100, 600)
    if data_type==0:
        for i in range(length):
            array.append(i/length)
    elif data_type==1:
        for i in range(length):
            array.append((length-i)/length)
    elif data_type==2:
        for i in range(length):
            array.append(np.random.rand())
    elif data_type==3:
        for i in range(length):
            if i<length/2:
                array.append(i/length)
            else:
                array.append((length - i) / length)
    elif data_type==4:
        for i in range(length):
            if i > length / 2:
                array.append(i / length)
            else:
                array.append((length - i) / length)
    array = noiser(array, noise)
    return array

def smooth(signal):
    out = np.abs(signal)
    return gaussian_filter1d(out, 3)

def get_envelope(self, signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

# fig, axs = plt.subplots(2,5)
# noise = 0.1
# for i in range(5):
#     number = np.random.randint(5)
#     number = i
#     data = data_generator(number)
#     axs[0,i].plot(smooth(crop_and_pad(data,noise)))
#     data = augment(data, 0.1,0.9)
#     axs[1][i].plot(smooth(crop_and_pad(data, noise)))
#     axs[0,i].set_title(number)
# plt.show()
#
#
# fig, axs = plt.subplots(5,5, sharex=True, sharey=True)
# noise = 0.1
# mapper = {0:'a',1:'b',2:'c',3:'d',4:'e'}
# for i in range(5):
#     for j in range(5):
#         data = data_generator(i)
#         axs[j,i].plot(smooth(crop_and_pad(data,noise)))
#         axs[0,i].set_title(mapper[i])
# plt.show()