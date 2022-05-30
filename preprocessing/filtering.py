import numpy as np
from heartpy import remove_baseline_wander
from heartpy.filtering import filter_signal

Fs = 300.0

def apply_filter(signal) -> np.array:
    denoised_signal = remove_noise(signal)
    return remove_baseline_wander(denoised_signal, sample_rate=Fs, cutoff=0.05)
    
def remove_noise(signal) -> np.array:
    low_band, high_band = 3, 45 
    return filter_signal(signal, cutoff = [low_band, high_band], sample_rate = Fs, order = 3, filtertype='bandpass')
