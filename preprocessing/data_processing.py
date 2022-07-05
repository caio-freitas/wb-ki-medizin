import pathlib
import numpy as np
import pandas as pd
from biosppy.signals import ecg
from utils import load_ecg, get_target
from .filtering import apply_filter
from .metrics import apply_metrics
from typing import Union
from scipy.io import savemat

import logging
# Create and configure logger
logger = logging.getLogger("meu_log")

def rr_peaks_from_ecg_signal(ecg_signal: pd.Series, sampling_freq=300) -> np.array:
    """Given an ECG signal function gets the RR peaks
    :param ecg_signal: ECG signal
    :param sampling_freq: sampling frequency
    :return: array with peak positions
    """
    fs = 300 # Sampling frequency (Hz)
    signal, r_peaks = ecg.ecg(ecg_signal, show=False, sampling_rate=sampling_freq)[1:3]

    rr = np.diff(r_peaks / 1000) # in seconds
    return rr

def add_noise(signal: pd.Series, std: float):
    '''Adds white gaussian noise of standar deviation std to signal
    '''
    noise = np.random.normal(0, std, signal.shape)
    return signal + noise
    
def generate_noisy_signals(original_signal: pd.Series, n: int, std: float):
    '''Creates multiple .mat files with random addition of noise on top of
    the original signal
    ''' 
    for i in range(n):
        noisy_signal = add_noise(ecg_signal, std)
        savemat(f"gen_{time.time()}.mat", {"val": [noisy_signal]})

def normalize(ecg_signal: np.array) -> np.array:
    """"Normalice de an ECG
        Introducing a vector of the electrocardigrom
        and it retuns the vector normalizated

    """
    avg, dev = ecg_signal.mean(), ecg_signal.std()
    return  (ecg_signal - avg) / dev


def pipeline(ecg_signal: np.array, sampling_freq: Union[int, float] = 300) -> np.array:
    """Process an ECG Signal
    :param ecg_signal: Electrocardiogram Signal
    :param sampling_freq: Sampling frequency
    :return: array with features of processed signal
    """
    # normalize data
    try:
        normalized_ecg = normalize(ecg_signal)
    except Exception as e:
        logging.error(f"Error normalizing data: {e}")
        normalized_ecg = ecg_signal

    # remove noise
    try:
        denoised_ecg = apply_filter(normalized_ecg)
    except Exception as e:
        logging.error(f"Error filtering data: {e}")
        denoised_ecg = normalized_ecg
    
    # TODO remove artifacts

    # extract rr peaks
    rr_peaks = rr_peaks_from_ecg_signal(denoised_ecg, sampling_freq)

    if len(rr_peaks) < 10:
        raise Exception(f"Cannot compute RR Peaks. Insuficient data for calculation: {len(rr_peaks)}")

    # extract metrics
    metrics = apply_metrics(rr_peaks, normalized_ecg)

    return metrics

def process_all_ecg() -> pd.DataFrame:
    """Load and transform data using Pipeline functions
    :returns: table which each line corresponds to one ECG signal and each column one metric of the signal
    """
    data = []

    # getting paths
    repo_dir = pathlib.Path(__file__).parent.parent
    training_dir = repo_dir / "training/"

    # applying functions in all files
    for filename in training_dir.iterdir():
        logging.debug(f"Processing: {filename.name}...")
        if filename.suffix == ".mat":
            ecg_signal = load_ecg(filename)
            target = get_target(filename.stem)
            try:
                processed_data = np.append(pipeline(ecg_signal), target)
                data.append(processed_data)
            except Exception as e:
                logger.debug(e)
                continue

    return np.array(data)
