# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title='Fault Detection System',
    page_icon='🔧',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better styling
st.markdown('''
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    h1 {
        color: #1f4788;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #ff6b6b;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 30px;
    }
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
''', unsafe_allow_html=True)

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

# Header
st.markdown('<h1>🔧 Industrial Machinery Fault Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style=\"font-size: 18px; color: #7f8c8d;\">Advanced Vibration Analysis using Band-Limited Pass Filter (BLPF) and Fast Fourier Transform (FFT)</p>', unsafe_allow_html=True)
st.markdown('---')

# Sidebar with better design
with st.sidebar:
    st.image('https://img.icons8.com/fluency/96/000000/settings.png', width=80)
    st.markdown('### ⚙️ Filter Parameters')
    st.markdown('---')
    
    fs = st.slider('📊 Sampling Frequency (Hz)', 1000, 20000, 10000, 100,
                   help='The rate at which vibration data is sampled')
    
    st.markdown('---')
    lowcut = st.slider('📉 Low Cutoff (Hz)', 10, 100, 20, 5,
                       help='Lower frequency bound for filtering')
    
    highcut = st.slider('📈 High Cutoff (Hz)', 500, 2000, 1000, 50,
                        help='Upper frequency bound for filtering')
    
    st.markdown('---')
    order = st.slider('🔢 Filter Order', 2, 10, 4, 1,
                      help='Higher order = sharper cutoff')
    
    st.markdown('---')
    st.info('💡 Adjust these parameters to fine-tune the fault detection sensitivity.')

# Load data
try:
    datafile = './data/vibration_dataset.csv'
    data = pd.read_csv(datafile)
    
    # Dataset Overview Section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('### 📊 Dataset Overview')
    with col2:
        st.metric('Total Samples', f'{data.shape[0]:,}')
    with col3:
        st.metric('Features', data.shape[1])
    
    # Display sample data in an expandable section
    with st.expander('🔍 View Dataset Sample (First 10 rows)', expanded=False):
        st.dataframe(data.head(10), use_container_width=True, height=300)
    
    st.markdown('---')
    
    # Feature Statistics
    features = ['mean', 'std', 'var', 'skewness', 'kurtosis', 'rms']
    available_features = [f for f in features if f in data.columns]
    
    st.markdown('### 📈 Statistical Analysis of Vibration Features')
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(['📊 Summary Statistics', '📉 Distributions', '🔬 Correlation Matrix'])
    
    with tab1:
        st.dataframe(data[available_features].describe(), use_container_width=True)
    
    with tab2:
        # Improved histograms
        fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig1.patch.set_facecolor('#f5f7fa')
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        for i, feature in enumerate(available_features[:6]):
            axes[i].hist(data[feature], bins=40, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[i].set_title(feature.upper(), fontsize=14, fontweight='bold', color='#2c3e50')
            axes[i].set_xlabel('Value', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].grid(True, alpha=0.3, linestyle='--')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with tab3:
        # Correlation heatmap
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = data[available_features].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        st.pyplot(fig_corr)
    
    st.markdown('---')
    
    # Signal Processing
    if 'rms' in data.columns:
        st.markdown('### 🔍 Frequency Spectrum Analysis')
        
        signal = data['rms'].values
        
        # Apply filter
        with st.spinner('Processing vibration signal...'):
            filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order)
            
            # FFT
            N = len(filtered_signal)
            fft_vals = np.fft.fft(filtered_signal)
            fft_freqs = np.fft.fftfreq(N, 1/fs)
            pos_mask = fft_freqs >= 0
            freqs = fft_freqs[pos_mask]
            amps = np.abs(fft_vals[pos_mask])
        
        # Key Metrics
        st.markdown('#### 🎯 Key Detection Metrics')
        col1, col2, col3, col4 = st.columns(4)
        
        peak_freq = freqs[np.argmax(amps)]
        max_amp = np.max(amps)
        mean_amp = np.mean(amps)
        rms_value = np.sqrt(np.mean(signal**2))
        
        with col1:
            st.markdown(f'''
                <div style=\"background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;\">
                    <h3 style=\"color: white; margin: 0;\">🎯 Peak Frequency</h3>
                    <h2 style=\"color: white; margin: 10px 0;\">{peak_freq:.2f} Hz</h2>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
                <div style=\"background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;\">
                    <h3 style=\"color: white; margin: 0;\">📊 Max Amplitude</h3>
                    <h2 style=\"color: white; margin: 10px 0;\">{max_amp:.2f}</h2>
                </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
                <div style=\"background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;\">
                    <h3 style=\"color: white; margin: 0;\">📈 Mean Amplitude</h3>
                    <h2 style=\"color: white; margin: 10px 0;\">{mean_amp:.2f}</h2>
                </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
                <div style=\"background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;\">
                    <h3 style=\"color: white; margin: 0;\">⚡ RMS Value</h3>
                    <h2 style=\"color: white; margin: 10px 0;\">{rms_value:.4f}</h2>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        # Enhanced spectrum plot
        fig2, ax = plt.subplots(figsize=(14, 7))
        fig2.patch.set_facecolor('#f5f7fa')
        
        ax.plot(freqs, amps, color='#667eea', linewidth=2, label='Frequency Spectrum')
        ax.fill_between(freqs, amps, alpha=0.3, color='#667eea')
        ax.axvline(peak_freq, color='#f5576c', linestyle='--', linewidth=2, 
                   label=f'Peak at {peak_freq:.2f} Hz')
        
        ax.set_title('Filtered RMS Vibration Spectrum', fontsize=18, fontweight='bold', 
                     color='#2c3e50', pad=20)
        ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='600')
        ax.set_ylabel('Amplitude', fontsize=14, fontweight='600')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Fault Detection Alert
        if peak_freq > 100:
            st.warning('⚠️ **Potential Fault Detected**: High-frequency vibrations detected. Recommend maintenance inspection.')
        else:
            st.success('✅ **System Status**: Vibration levels within normal range.')
    
    st.markdown('---')
    st.markdown('<p style=\"text-align: center; color: #95a5a6;\">© 2025 Fault Detection System | Developed by Lokesh Kumar</p>', unsafe_allow_html=True)
    
except Exception as e:
    st.error(f'❌ Error loading data: {str(e)}')
    st.info('Please ensure vibration_dataset.csv is in the data/ folder.')
