import matplotlib.pyplot as plt
from scipy import signal
import librosa.display
import numpy as np
import librosa

# Load from file
control = 'data/examples/DataN/corncontrol1.wav'
control_signal, control_sample_rate = librosa.load(control)

treatment = 'data/examples/DataN/corntreatment1.wav'
treatment_signal, treat_sample_rate = librosa.load(treatment)

# Plot the signal
plt.plot(control_signal, label='control')
plt.plot(treatment_signal, label='treatment')
plt.xticks()
plt.legend()
plt.show()

# Plot the FFT
y = np.fft.fft(control_signal)
freq = np.fft.fftfreq(len(control_signal), 1 / control_sample_rate)
plt.plot(abs(freq), np.abs(y))

# Notch filter for US mains freq
exclusion_freqs = [60, 120, 180, 240, 300, 360, 420]
fs = control_sample_rate  # Sample frequency (Hz)
Q = 30.0  # Quality factor
out_control = control_signal.copy()
out_treat = treatment_signal.copy()
for f0 in exclusion_freqs:
    b, a = signal.iirnotch(f0, Q, fs)
    out_control = signal.filtfilt(b, a, out_control)
    out_treat = signal.filtfilt(b, a, out_treat)

# Compare signals before/after notch filter
fig, axs = plt.subplots(2,2)
ymin, ymax = min(min(control_signal), min(out_control)), max(max(control_signal), max(out_control))
axs[0,0].plot(control_signal[1000:2000])
axs[0,0].set_ylim(-0.0005, 0.0005)
axs[0,0].set_title('Control signal, unfiltered')
axs[0,1].plot(out_control[1000:2000])
axs[0,1].set_ylim(-0.0005, 0.0005)
axs[0,1].set_title('Control signal, filtered')
ymin, ymax = min(min(treatment_signal), min(out_treat)), max(max(treatment_signal), max(out_treat))
axs[1,0].plot(treatment_signal[23627000:23628000])
axs[1,0].set_ylim(-0.0005, 0.0005)
axs[1,0].set_title('Treatment signal, filtered')
axs[1,1].plot(out_treat[23627000:23628000])
axs[1,1].set_ylim(-0.0005, 0.0005)
axs[1,1].set_title('Treatment signal, filtered')
plt.tight_layout()

# Plot a spectrogram of the result
stft = librosa.stft(control_signal[1000:2000]/0.001)
stft_mag, _ = librosa.magphase(stft)
plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.amplitude_to_db(stft_mag, ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

