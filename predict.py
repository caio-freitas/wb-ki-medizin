import joblib
import pandas as pd
import numpy as np
import pathlib
from preprocessing import pipeline
from preprocessing.metrics import feature_names
from utils import load_ecg

def main():
    # Loads model and file to be predicted
    model = joblib.load('lgbm.pkl')

    test_data, filenames = process_test_data()

    df = pd.DataFrame(test_data, columns=feature_names)
    
    predictions = pd.DataFrame(model.predict(df), columns=["prediction"])
    predictions["filename"] = filenames

    predictions.to_csv("predictions.csv", index=False)

def process_test_data():
    data = []
    filenames = []

    repo_dir = pathlib.Path(__file__).parent
    test_dir = repo_dir / "test/"

    # applying functions in all files
    for filename in test_dir.iterdir():
        print(f"Processing: {filename.name}...")
        if filename.suffix == ".mat":
            ecg_signal = load_ecg(filename)
            try:
                processed_data = pipeline(ecg_signal)
                filenames.append(filename.name)
                data.append(processed_data)
            except Exception as e:
                print(e)
                continue

    return np.array(data), filenames

if __name__ == "__main__":
    main()
