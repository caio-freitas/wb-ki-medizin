import pathlib
from preprocessing import process_all_ecg
from metrics import col_names
from utils import csv_export


if __name__ == "__main__":
    data = process_all_ecg()
    csv_export(
        data=data, 
        path=pathlib.Path(__file__).parent, 
        name="features.csv",
        cols=col_names
    )
    