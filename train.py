import sys
import joblib
import pathlib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

from preprocessing import process_all_ecg
from preprocessing.metrics import feature_names, col_names
from utils import csv_export


def main():
    
    model_filename = 'lgbm.pkl'
    # Runs data pipeline
    if not (pathlib.Path(__file__).parent / "features.csv").exists():
        data = process_all_ecg()
        csv_export(
            data=data, 
            path=pathlib.Path(__file__).parent, 
            name="features.csv",
            cols=col_names
        )
    
    # Read resulting dataframe
    df = pd.read_csv("features.csv")

    y_train = df["label"]
    X_train = df.drop(["label"], axis=1)

    model = LGBMClassifier(
        metric="multi_logloss",
        num_leaves=32
    )

    model.fit(X_train, y_train)

    joblib.dump(model, model_filename)

    print(f"Saved model {model_filename}")

if __name__ == "__main__":
    main()