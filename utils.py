import pandas as pd
import numpy as np
import pathlib
from scipy.io import loadmat
from typing import List

def load_ecg(filepath: pathlib.Path) -> pd.Series:
    """Function to load ECG signal as pandas Series
    :param filepath: papth of the ECG signal file
    :return: ECG signal as Series
    """

    ecg = pd.Series(loadmat(filepath)["val"][0])
    return ecg


def get_target(filenamestem: str):
    repo_dir = pathlib.Path(__file__).parent
    training_dir = repo_dir / "../training/"
    reference_df = pd.read_csv(training_dir / "REFERENCE.csv", names=["filename", "label"])
    return reference_df[reference_df["filename"] == filenamestem]["label"].values[0]

