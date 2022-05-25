import sys
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from data_processing import pipeline
from utils import load_ecg

def main():
    # Loads model and file to be predicted
    if len(sys.argv) > 1:
        file_to_predict = sys.argv[1]
        
        if len(sys.argv) > 2:
            model = joblib.load(sys.argv[2])
            return
        else: 
            model = joblib.load('lgbm.pkl')
    else: 
        file_to_predict = "train_ecg_04085.mat"

    ecg_signal = load_ecg(file_to_predict)
    data = [pipeline(ecg_signal)]
    df = pd.DataFrame(data, columns=["min_rate", "avg_rate", "max_rate", "sdnn", "nn50", "sdsd", "rmssd"])
    
    predictions = pd.DataFrame(model.predict(df))

    predictions.to_csv("predictions.csv")

if __name__ == "__main__":
    main()
