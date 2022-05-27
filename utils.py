import pandas as pd
import numpy as np
import pathlib
from scipy.io import loadmat

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

def get_target(filenamestem: str):
    repo_dir = pathlib.Path(__file__).parent
    training_dir = repo_dir / "training/"
    reference_df = pd.read_csv(training_dir / "REFERENCE.csv", names=["filename", "label"])
    return reference_df[reference_df["filename"] == filenamestem]["label"].values[0]


def csv_export(data: np.array, path: pathlib.Path = None, name: str = None) -> None:
    """Export numpy array as CSV
    :param data: array to be exported
    :param path: path to be exported
    :param name: name of the file
    """
    df = pd.DataFrame(data, columns=["min_rate", "avg_rate", "std_rate","max_rate", "sdnn", "nn50", "sdsd", "rmssd", "label"])
    df.to_csv(path / name, index=False)
    # new_name = (get_target(name[:-4]) + "_" + name)
    # pd.Series(data).to_csv(path / new_name, index=False)


