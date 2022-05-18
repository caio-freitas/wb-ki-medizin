import pathlib
from utils import load_ecg, csv_export, rr_peaks_from_ecg_signal


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
        

if __name__ == '__main__':
    process_training_set()
