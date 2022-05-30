import pathlib
import numpy as np
import pandas as pd
from biosppy.signals import ecg
from utils import load_ecg, get_target
from .filtering import apply_filter
from .metrics import apply_metrics

def rr_peaks_from_ecg_signal(ecg_signal: pd.Series) -> np.array:
    """Given an ECG signal function gets the RR peaks
    :param ecg_signal: ECG signal
    :return: array with peak positions
    """
    fs = 300 # Sampling frequency (Hz)
    signal, r_peaks = ecg.ecg(ecg_signal, show=False, sampling_rate=300)[1:3]

    rr = np.diff(r_peaks / 1000) # in seconds
    return rr


def normalize(ecg_signal: np.array) -> np.array:
    """"Normalice de an ECG
        Introducing a vector of the electrocardigrom
        and it retuns the vector normalizated

    """
    avg, dev = ecg_signal.mean(), ecg_signal.std()
    return  (ecg_signal - avg) / dev


def pipeline(ecg_signal: np.array) -> np.array:
    """Process an ECG Signal
    """
    # normalize data
    normalized_ecg = normalize(ecg_signal)

    # remove noise
    denoised_ecg = apply_filter(normalized_ecg)
    
    # TODO remove artifacts

    # extract rr peaks
    rr_peaks = rr_peaks_from_ecg_signal(denoised_ecg)

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
        print(f"Processing: {filename.name}...")
        if filename.suffix == ".mat":
            ecg_signal = load_ecg(filename)
            target = get_target(filename.stem)
            try:
                processed_data = np.append(pipeline(ecg_signal), target)
                data.append(processed_data)
            except Exception as e:
                print(e)
                continue

    return np.array(data)
