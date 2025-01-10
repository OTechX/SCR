import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.signal import find_peaks

# parameter settings
fs = 48000
sync_len = 500
preamble_blank = 4800

# pilot signal
def load_sync():
    sync_1 = np.fromfile('./data/sync.dat','int16')
    sync_2 = sync_1 / abs(sync_1).max()
    plt.plot(sync_2)
    return sync_2

# sensing signal
def load_signal_ear():
    base_1 = np.fromfile('./data/signal_ear.dat','int16')
    base_2 = base_1[0:1200]
    base_3 = base_2 / abs(base_2).max()#归一化
    plt.plot(base_3)
    return base_3

# find start point of sensing process
def find_point(signal, sync, para):
    corr = np.correlate(signal[:15000], sync, "same")#peak in pilot
    corrnorm = corr / abs(corr).max()
    sigpeaks, _ = find_peaks(np.abs(corrnorm), height=(para,1))
    return sigpeaks[2]+ 1.5*sync_len + preamble_blank

# finer synchronization,segmentation
def seg(signal, base):
    corr = np.correlate(signal, base, "same")
    peak_index = np.argmax(np.abs(corr))
    seg = signal[peak_index-600:peak_index+600]
    return seg

# bandpass filter
def bandfilt(signal):
    FS = 48000
    wn1 = 17000 / (FS / 2)
    wn2 = 23000 / (FS / 2)
    sos = butter(8, [wn1, wn2], btype='bandpass', output='sos')
    y = sosfilt(sos, signal)
    return y


