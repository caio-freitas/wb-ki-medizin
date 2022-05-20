import pathlib
import pandas as pd
from utils import load_ecg, csv_export, rr_peaks_from_ecg_signal, apply_metrics


def process_training_set():
    """Apply some utils funtions in all the training set files
    """
    # getting paths
    repo_dir = pathlib.Path(__file__).parent
    training_dir = repo_dir.parent / "training/"
    save_dir = repo_dir.parent / "rr_training/"
    if not save_dir.exists():
        save_dir.mkdir()

    # applying functions in all files
    for data in training_dir.iterdir():
        if data.suffix == ".mat":
            ecg_signal = load_ecg(data.name)
            rr_peaks = rr_peaks_from_ecg_signal(ecg_signal)
            csv_export(rr_peaks, save_dir, data.stem + ".csv")
        

    def process_features():
    """Calculate features for csv files and return it into a final csv
    """
    repo_dir = pathlib.Path(__file__).parent
    rr_dir = repo_dir / "rr_training/"

    df = pd.DataFrame()

    for data in rr_dir.iterdir():
        d = pd.read_csv(data).to_numpy()
        print(d)
        try:
        df = pd.concat([df, apply_metrics(d)])
        except Exception as e:
        print(e)

    df.to_csv("features.csv")


if __name__ == '__main__':
    # process_training_set()
    process_features()
