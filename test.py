import utils
import pandas as pd



if __name__ == '__main__':

    ecg_signal = utils.load_ecg("train_ecg_00001.mat")
    rr_peaks = utils.rr_peaks_from_ecg_signal(ecg_signal)

    df = utils.apply_metrics(rr_peaks)
    print(df)