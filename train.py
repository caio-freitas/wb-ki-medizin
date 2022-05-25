import sys
import joblib
import pathlib
import pandas as pd
from regex import R
from data_processing import process_all_ecg
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import csv_export

def main():
    
    if len(sys.argv) > 1:
        model_filename = sys.argv[1]
        if len(sys.argv) > 2:
            return
    else: 
        model_filename = 'lgbm.pkl'
    # Runs data pipeline
    # data = process_all_ecg()
    # csv_export(
    #     data=data, 
    #     path=pathlib.Path(__file__).parent, 
    #     name="features.csv"
    # )

    # Read resulting dataframe
    df = pd.read_csv("features.csv")
    df.head()
    df = df[df["label"] != "~"]


    X_train, X_test, y_train, y_test = train_test_split(df[['min_rate', 'avg_rate', 'max_rate', 'sdnn', 'nn50', 'sdsd', 'rmssd']],
                                                        df[["label"]])
    print("Dataset splitted: {} training samples | {} test samples".format(X_train.size, X_test.size))

    model = LGBMClassifier(
        metric="multi_logloss",
        num_leaves=32
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = f1_score(y_test, y_pred, average="weighted")

    print("Final model score: {}".format(score))

    joblib.dump(model, model_filename)

    print("Saved model  {}".format(model_filename))

if __name__ == "__main__":
    main()