import pandas as pd
import numpy as np
import pathlib
from scipy.io import loadmat
from ecgdetectors import Detectors
from pyhrv.time_domain import sdnn, sdann, nn50, sdsd, tinn, rmssd

def load_ecg(filename: str) -> pd.Series:
    """Function to load ECG signal as pandas Series
    :param filename: name of the ECG signal file
    :return: ECG signal as Series
    """
    # get path of the training/ directory
    repo_path = pathlib.Path(__file__).parent
    training_set_path = repo_path / "training/"
    ecg = pd.Series(loadmat(training_set_path / filename)["val"][0])
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
    
    
def normalize(ecg_signal: np.array) -> np.array:
    """"Normalice de an ECG
        Introducing a vector of the electrocardigrom
        and it retuns the vector normalizated

    """
    avg, dev = ecg_signal.mean(), ecg_signal.std()
    return  (ecg_signal - avg) / dev


def get_target(filenamestem: str):
    repo_dir = pathlib.Path(__file__).parent
    training_dir = repo_dir / "training/"
    reference_df = pd.read_csv(training_dir / "REFERENCE.csv", names=["filename", "label"])
    return reference_df[reference_df["filename"] == filenamestem]["label"].values[0]


def csv_export(data: np.array, path: pathlib.Path, name: str) -> None:
    """Export numpy array as CSV
    :param data: array to be exported
    :param path: path to be exported
    :param name: name of the file
    """
    new_name = (get_target(name[:-4]) + "_" + name)
    pd.Series(data).to_csv(path / new_name, index=False)
    
    
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
    
    return pd.DataFrame.from_dict({
        "SDNN": sdnn(peaks),
        "SDANN": sdann(peaks),
        "NN50": sdnn(peaks),
        "SDSD": sdnn(peaks),
        "TINN": sdnn(peaks),
        "r-MSSD": rmssd(peaks)
    })
