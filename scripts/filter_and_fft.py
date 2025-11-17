# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Parameters
datafile = "./data/vibration_dataset.csv"
fs = 10000.0
lowcut = 20.0
highcut = 1000.0
order = 4

# Load dataset
data = pd.read_csv(datafile)
# Debug: print exact loaded column names
print("Columns in data:", data.columns.tolist())
print(data.head())

# Strip whitespace in column names if any
data.columns = data.columns.str.strip()

features = ["mean", "std", "var", "skewness", "kurtosis", "rms", "peaktopeak",
            "crestfactor", "impulsefactor", "shapefactor", "entropy"]

# Check if all features exist in columns
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"Warning: These features are missing from the data and will be skipped: {missing_features}")

available_features = [f for f in features if f in data.columns]

print("\nSummary statistics of available features:")
print(data[available_features].describe())

# Plot histograms
plt.figure(figsize=(15, 10))
for i, feature in enumerate(available_features, 1):
    plt.subplot(3, 4, i)
    plt.hist(data[feature], bins=30, color='c', alpha=0.75)
    plt.title(feature)
    plt.tight_layout()
plt.show()

# Use 'rms' if available for filtering and FFT
if "rms" in data.columns:
    signal = data["rms"].values
else:
    raise ValueError("Column 'rms' not found in data.")

filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order)

N = len(filtered_signal)
fft_vals = np.fft.fft(filtered_signal)
fft_freqs = np.fft.fftfreq(N, 1/fs)
pos_mask = fft_freqs >= 0
freqs = fft_freqs[pos_mask]
amps = np.abs(fft_vals[pos_mask])

print("\nSample Frequency (Hz):", freqs[:10])
print("Sample Amplitude:", amps[:10])

plt.figure(figsize=(12,6))
plt.plot(freqs, amps, label='Filtered RMS Spectrum')
plt.title('Filtered RMS Vibration Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
