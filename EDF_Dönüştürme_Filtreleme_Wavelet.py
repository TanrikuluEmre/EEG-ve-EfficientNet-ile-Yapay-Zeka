import os
import pyedflib
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pywt

eeg_directory = 'EEG\\'
spectrogram_directory = 'spectrograms\\'
wavelet_directory = 'wavelet_transforms\\'
os.makedirs(spectrogram_directory, exist_ok=True)
os.makedirs(wavelet_directory, exist_ok=True)

eeg_files = [file for file in os.listdir(eeg_directory) if file.endswith('.edf')]

for eeg_file in eeg_files:
    try:
        edf_file = pyedflib.EdfReader(os.path.join(eeg_directory, eeg_file))
        num_channels = edf_file.signals_in_file

        for channel in range(num_channels):
            eeg_data = edf_file.readSignal(channel)
            
            fs = edf_file.getSampleFrequency(channel)
            nyquist = 0.5 * fs
            low = 4 / nyquist
            high = 30 / nyquist
            b, a = scipy.signal.butter(4, [low, high], btype='band')

            filtered_data = scipy.signal.lfilter(b, a, eeg_data)

            # Spektrogram
            plt.specgram(filtered_data, Fs=fs, NFFT=256, noverlap=128, cmap='viridis')

            plt.xlabel('Zaman (s)')
            plt.ylabel('Frekans (Hz)')
            plt.title(f'EEG Spektrogram - Dosya: {eeg_file} - Kanal {channel + 1}')
            plt.colorbar(label='Güç (dB)')

            save_path_spectrogram = os.path.join(spectrogram_directory, f'{os.path.splitext(eeg_file)[0]}_channel{channel + 1}_spektrogram.png')
            plt.savefig(save_path_spectrogram)
            plt.close()

            # Wavelet Transform
            scales = np.arange(4,30)
            coeffs, freqs = pywt.cwt(filtered_data, scales, 'cmor')

            plt.pcolormesh(np.arange(len(filtered_data)), freqs, np.abs(coeffs), shading='auto')
            plt.xlabel('Zaman (örnek)')
            plt.ylabel('Ölçek')
            plt.title(f'EEG Wavelet Transform - Dosya: {eeg_file} - Kanal {channel + 1}')
            plt.colorbar(label='Magnitude')

            save_path_wavelet = os.path.join(wavelet_directory, f'{os.path.splitext(eeg_file)[0]}_channel{channel + 1}_wavelet.png')
            plt.savefig(save_path_wavelet)
            plt.close()

        edf_file.close()

    except Exception as e:
        print(f"Hata oluştu: {e}")
        continue
