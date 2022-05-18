import pandas as pd
import numpy as np
import pathlib
from scipy.io import loadmat
from ecgdetectors import Detectors


def load_ecg(filename: str) -> pd.Series:
    """Function to load ECG signal as pandas Series
    :param filename: name of the ECG signal file
    :return: ECG signal as Series
    """
    # get path of the training/ directory
    repo_path = pathlib.Path(__file__).parent
    training_set_path = repo_path.parent / "training/"
    try: 
        ecg = pd.Series(loadmat(training_set_path / filename)["val"][0])
    except Exception as e:
        print(e)

    return ecg


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


def csv_export(data: np.array, path: pathlib.Path, name: str) -> None:
    """Export numpy array as CSV
    :param data: array to be exported
    :param path: path to be exported
    :param name: name of the file
    """
    pd.Series(data).to_csv(path / name)
    