import numpy as np
from scipy import signal


# parameter settings
fs = 48000
window = 'hann'
nperseg = 256
noverlap = 250
nfft = 256


#magnitude
def extract_psd(onedata, window, nperseg, noverlap, nfft):
    f, t, Sxx = signal.spectrogram(onedata, fs, window, nperseg, noverlap, nfft, 
            detrend='constant', return_onesided=True,scaling='density', 
            axis=-1, mode='psd')
    Sxx_dB = 10*np.log10(Sxx)
    Sxx_dB_refine = Sxx_dB[64:]
    return Sxx_dB_refine


#phase
def extract_phase(onedata, window, nperseg, noverlap, nfft):
    f, t, Sxx_phase = signal.spectrogram(onedata, fs, window, nperseg, noverlap, nfft, 
            detrend='constant', return_onesided=True,scaling='density', 
            axis=-1, mode='phase')
    Sxx_phase_refine = Sxx_phase[64:]
    return Sxx_phase_refine


# normalization
def norm(nn):
    nn[np.where(nn < 0.1 * nn.std())] = 0.1 * nn.std()
    return (nn - nn.min()) / (nn.max() - nn.min())


# differential
def differential(seg_s, seg_r):
    mag_r = extract_psd(seg_r)
    phase_r = extract_phase(seg_r)
    mag_s = extract_psd(seg_s)
    phase_s = extract_phase(seg_s)
    diff_mag = abs(mag_s - mag_r)
    diff_phase = abs(phase_s - phase_r)
    diff_mag_norm = norm(diff_mag)
    diff_phase_norm = norm(diff_phase)
    spec = np.array([diff_mag_norm,diff_phase_norm])
    return spec


