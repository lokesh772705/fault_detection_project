# -*- coding: utf-8 -*-
import streamlit as st
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

# Streamlit App
st.title('🔧 Fault Detection in Industrial Machinery')
st.subheader('Vibration Analysis using BLPF and FFT')

# Sidebar parameters
st.sidebar.header('Filter Parameters')
fs = st.sidebar.slider('Sampling Frequency (Hz)', 1000, 20000, 10000)
lowcut = st.sidebar.slider('Low Cutoff (Hz)', 10, 100, 20)
highcut = st.sidebar.slider('High Cutoff (Hz)', 500, 2000, 1000)
order = st.sidebar.slider('Filter Order', 2, 10, 4)

# Load data
datafile = './data/vibration_dataset.csv'
data = pd.read_csv(datafile)

# Dataset overview
st.header('📊 Dataset Overview')
st.write(f'Dataset shape: {data.shape}')
st.dataframe(data.head(10))

# Summary statistics
features = ['mean', 'std', 'var', 'skewness', 'kurtosis', 'rms']
available_features = [f for f in features if f in data.columns]

st.header('📈 Feature Statistics')
st.dataframe(data[available_features].describe())

# Histograms
st.header('📉 Feature Distributions')
fig1, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for i, feature in enumerate(available_features[:6]):
    axes[i].hist(data[feature], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    axes[i].set_title(feature.upper())
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

# Signal processing
if 'rms' in data.columns:
    st.header('🔍 Vibration Spectrum Analysis')
    signal = data['rms'].values
    
    # Apply filter
    filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order)
    
    # FFT
    N = len(filtered_signal)
    fft_vals = np.fft.fft(filtered_signal)
    fft_freqs = np.fft.fftfreq(N, 1/fs)
    pos_mask = fft_freqs >= 0
    freqs = fft_freqs[pos_mask]
    amps = np.abs(fft_vals[pos_mask])
    
    # Plot spectrum
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, amps, color='darkblue', linewidth=1.5)
    ax.set_title('Filtered RMS Vibration Spectrum', fontsize=16, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # Display key metrics
    st.subheader('Key Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric('Peak Frequency', f'{freqs[np.argmax(amps)]:.2f} Hz')
    col2.metric('Max Amplitude', f'{np.max(amps):.2f}')
    col3.metric('Mean Amplitude', f'{np.mean(amps):.2f}')
    
st.success('✅ Analysis Complete!')
