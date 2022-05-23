import pathlib
import numpy as np
import pandas as pd
from ecgdetectors import Detectors
from pyhrv.time_domain import sdnn, sdann, nn50, sdsd, tinn, rmssd
from utils import load_ecg, csv_export, get_target

def rr_peaks_from_ecg_signal(ecg_signal: pd.Series) -> np.array:
    """Given an ECG signal function gets the RR peaks
    :param ecg_signal: ECG signal
    :return: array with peak positions ??? TODO verify that
    """
    fs = 300 # Sampling frequency
    detectors = Detectors(fs)

    r_peaks = detectors.hamilton_detector(ecg_signal) # RR distance detection
    
    rr = np.diff(r_peaks) / fs * 1000
    return rr


def normalize(ecg_signal: np.array) -> np.array:
    """"Normalice de an ECG
        Introducing a vector of the electrocardigrom
        and it retuns the vector normalizated

    """
    avg, dev = ecg_signal.mean(), ecg_signal.std()
    return  (ecg_signal - avg) / dev


def apply_metrics(peaks: np.array) -> pd.DataFrame:
    """Given the RR peaks array, returns a DataFrame
    with all features of interess
    
    * SDNN: Standard deviation of RR intervals series
    * SDANN: Standard deviation of the mean of RR intervals in 5-min segments
    * pNN50: Proportion of adjacent RR intervals differing by more than 50 ms
    * SDSD: Standard deviation of differences between adjacent RR intervals
    * TINN: Baseline width of the triangular interpolation of the intervals histogram
    * r-MSSD: The square root of the mean of the squares of differences between adjacent RR intervals
    """
    
    return np.array([
        sdnn(peaks)["sdnn"],
        #sdann(peaks)["sdann"], # warnings.warn("Signal duration too short for SDANN computation.")
        nn50(peaks)["nn50"],
        sdsd(peaks)["sdsd"],
        #tinn(peaks, plot=False)["tinn"], # warnings.warn('CAUTION: The TINN computation is currently providing incorrect results in the most cases due to a 
        rmssd(peaks)["rmssd"]
    ])


def pipeline(ecg_signal: np.array) -> np.array:
    """Process an ECG Signal
    """
    # normalize data
    normalized_ecg = normalize(ecg_signal)

    # remove noise
    #ohne_noise_ecg = remove_noise(normalized_ecg)
    
    # TODO remove artifacts

    # extract rr peaks
    rr_peaks = rr_peaks_from_ecg_signal(normalized_ecg)

    # extract metrics
    metrics = apply_metrics(rr_peaks)

    return metrics


def process_all_ecg() -> pd.DataFrame:
    """Load and transform data using Pipeline functions
    :returns: table which each line corresponds to one ECG signal and each column one metric of the signal
    """
    data = []

    # getting paths
    repo_dir = pathlib.Path(__file__).parent
    training_dir = repo_dir / "training/"

    # applying functions in all files
    for filename in training_dir.iterdir():
        if filename.suffix == ".mat":
            ecg_signal = load_ecg(filename.name)
            target = get_target(filename.stem)
            processed_data = pipeline(ecg_signal).append(target)
            data.append(processed_data)
            print(f"Processing: {filename.name}...")

    return np.array(data)

def process_one_ecg():
    ecg_signal = load_ecg("train_ecg_00001.mat")

    data = pipeline(ecg_signal)
    csv_export(data)


if __name__ == "__main__":
    data = process_all_ecg()
    csv_export(data)
    
