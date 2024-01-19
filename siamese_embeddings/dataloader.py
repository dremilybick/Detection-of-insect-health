import random

import torch
import numpy as np
from torch.utils.data import Dataset
from augment import augment, crop_and_pad, data_generator, smooth, get_envelope
import torch.nn.functional as F
import torch
from torch.utils.data import Sampler
import librosa


class ContrastiveSampler(Sampler):
    def __init__(self, siamese_dataset, batch_size=32, margin=1.0):
        self.siamese_dataset = siamese_dataset
        self.margin = margin
        self.batch_size = batch_size

    def __iter__(self):
        num_samples = len(self.siamese_dataset)
        indices = torch.randperm(num_samples).tolist()
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            batch = []
            for anchor_index in batch_indices:
                negative_index = anchor_index
                while self.siamese_dataset.origin[negative_index]==self.siamese_dataset.origin[anchor_index]:
                    negative_index = self._get_negative_index()
                batch.append([anchor_index, anchor_index, negative_index])

            yield batch

    def __len__(self):
        return len(self.siamese_dataset)

    def _get_negative_index(self):
        return torch.randint(0, len(self.siamese_dataset), (1,)).item()


class SiameseGeneratedDataset(Dataset):
    def __init__(self, signals, thresholds, origins):
        self.signals = signals
        self.thresholds = thresholds
        self.origin = origins

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        anchor = data_generator(np.random.randint(5), 0.1)
        negative = data_generator(np.random.randint(5),0.1)
        signal = [anchor, anchor, negative]
        noise_level = [0.1,0.1,0.1]

        signal[0] = crop_and_pad(signal[0], noise_level[0])

        signal[1] = augment(signal[1], noise_level[1], 0.9)
        signal[1] = crop_and_pad(signal[1], noise_level[1])

        signal[2] = crop_and_pad(signal[2], noise_level[2])

        signal = [smooth(x) for x in signal]

        sample = torch.FloatTensor(np.array(signal))

        return [sample[0],sample[1],sample[2]]

class SiameseDataset(Dataset):
    def __init__(self, signals, thresholds, origins):
        self.signals = signals
        self.thresholds = thresholds
        self.origin = origins

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal = np.array(self.signals[index])
        # signal = [self.get_envelope(x) for x in signal]

        noise_level = self.thresholds[index]
        noise_level = [x for x in noise_level]

        signal[0] = crop_and_pad(signal[0], noise_level[0])

        signal[1] = augment(signal[1], noise_level[1], 0.9)
        signal[1] = crop_and_pad(signal[1], noise_level[1])

        signal[2] = crop_and_pad(signal[2], noise_level[2])

        signal = [smooth(x) for x in signal]

        sample = torch.FloatTensor(np.array(signal))

        return [sample[0],sample[1],sample[2]]


class SiamesePredictionDataset(Dataset):
    def __init__(self, signals, thresholds):
        self.signals = signals
        self.thresholds = thresholds

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal = np.array(self.signals[index])
        noise_level = self.thresholds[index]

        signal = crop_and_pad(signal, noise_level)

        signal = smooth(signal)

        sample = torch.FloatTensor(np.array(signal))

        return sample



class SiameseSpectralDataset(Dataset):
    def __init__(self, signals, thresholds, origins):
        self.signals = signals
        self.thresholds = thresholds
        self.origin = origins
        self.n_mfcc = 13

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal = np.array(self.signals[index])

        noise_level = self.thresholds[index]
        noise_level = [x/20 for x in noise_level]

        signal[0] = crop_and_pad(signal[0], noise_level[0])
        signal[0] = self.generate_spectrogram(signal[0])

        signal[1] = augment(signal[1], noise_level[1], 0)
        signal[1] = crop_and_pad(signal[1], noise_level[1])
        signal[1] = self.generate_spectrogram(signal[1])

        signal[2] = crop_and_pad(signal[2], noise_level[2])
        signal[2] = self.generate_spectrogram(signal[2])

        sample = [torch.from_numpy(np.array(elem)).to(dtype=torch.float32) for elem in signal]

        return sample

    def generate_spectrogram(self, signal):
        hop_length = int(len(signal) / (self.n_mfcc + 1))  # Adjust hop_length based on n_mfcc
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=256, hop_length=hop_length)), ref=np.max)
        return spectrogram


class SiameseSpectralPredictionDataset(Dataset):
    def __init__(self, signals, thresholds):
        self.signals = signals
        self.thresholds = thresholds
        self.n_mfcc = 13

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal = np.array(self.signals[index])
        noise_level = self.thresholds[index]

        signal = crop_and_pad(signal, noise_level/20)
        signal = self.generate_spectrogram(signal)

        sample = torch.from_numpy(np.array(signal)).to(dtype=torch.float32)

        return sample

    def generate_spectrogram(self, signal):
        hop_length = int(len(signal) / (self.n_mfcc + 1))  # Adjust hop_length based on n_mfcc
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=256, hop_length=hop_length)), ref=np.max)
        return spectrogram