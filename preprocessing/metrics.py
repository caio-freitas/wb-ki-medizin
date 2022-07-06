import scipy
import numpy as np
import pandas as pd
from biosppy.signals import ecg
from pyhrv.time_domain import sdnn, sdann, nn50, sdsd, tinn, rmssd
from typing import Union

import logging
logger = logging.getLogger("main_log")

feature_names = [
    "min_rate",
    "avg_rate", 
    "std_rate",
    "max_rate", 
    "sdnn", 
    "nn50", 
    "sdsd", 
    "rmssd", 
    "low_freq_power_perc", 
    "high_freq_power_perc", 
    "freq_power_ratio"
]

col_names = feature_names + ["label"]

def rr_hr_from_ecg_signal(ecg_signal: pd.Series, sampling_freq=300) -> np.array:
    """Given an ECG signal function gets the RR peaks
    :param ecg_signal: ECG signal
    :param sampling_freq: sampling frequency
    :return: array with peak positions
    """
    fs = 300 # Sampling frequency (Hz)
    ecg_proc = ecg.ecg(ecg_signal, show=False, sampling_rate=sampling_freq)
    signal, r_peaks = ecg_proc[1:3]
    heart_rate = ecg_proc[6]

    rr = np.diff(r_peaks / 1000) # in seconds
    return rr, heart_rate

def spectral_powers(signal: np.array, LF: np.array = [0.05, 0.15], HF: np.array = [0.15, 0.4]):
    psd_f, psd = scipy.signal.welch(signal) # power spectral density

    psd_f_lf = psd_f[(psd_f > LF[0]) & (psd_f <= LF[1])]
    psd_lf = psd[(psd_f > LF[0]) & (psd_f <= LF[1])]
    
    psd_f_hf = psd_f[(psd_f > HF[0]) & (psd_f <= HF[1])]
    psd_hf = psd[(psd_f > HF[0]) & (psd_f <= HF[1])]

    total_power = np.trapz(psd, psd_f)

    LF_power = np.trapz(psd_lf, psd_f_lf) # low frequency band
    HF_power = np.trapz(psd_hf, psd_f_hf) # high frequency band

    return LF_power/total_power, HF_power/total_power, LF_power/HF_power

def apply_metrics(signal: np.array, sampling_freq: Union[int, float]) -> pd.DataFrame:
    """Given the RR peaks array, returns a DataFrame
    with all features of interess
    
    * heart_rate: Heart rate
    * SDNN: Standard deviation of RR intervals series
    * SDANN: Standard deviation of the mean of RR intervals in 5-min segments
    * pNN50: Proportion of adjacent RR intervals differing by more than 50 ms
    * SDSD: Standard deviation of differences between adjacent RR intervals
    * TINN: Baseline width of the triangular interpolation of the intervals histogram
    * r-MSSD: The square root of the mean of the squares of differences between adjacent RR intervals

    **** add/remove new metrics also in csv_export() at utils.py
    
    """

    # extract rr peaks
    rr_peaks, hrp = rr_hr_from_ecg_signal(signal, sampling_freq)
    
    if not hrp.any():
        logger.warning("Heart rate array is empty! Replacing with 0")
        hrp = np.array([0])
    
    [LF_power, HF_power, ratio] = spectral_powers(signal)
    
    return np.array([
        hrp.min(),
        hrp.mean(),
        hrp.std(),
        hrp.max(),
        sdnn(rr_peaks)["sdnn"],
        #sdann(rr_peaks)["sdann"], # warnings.warn("Signal duration too short for SDANN computation.")
        nn50(rr_peaks)["nn50"],
        sdsd(rr_peaks)["sdsd"],
        #tinn(rr_peaks, plot=False)["tinn"], # warnings.warn('CAUTION: The TINN computation is currently providing incorrect results in the most cases due to a 
        rmssd(rr_peaks)["rmssd"],
        LF_power,
        HF_power,
        ratio
    ])
