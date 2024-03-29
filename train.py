import pathlib
import joblib
import logging
import pandas as pd

from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from preprocessing import pipeline
from preprocessing.metrics import feature_names
from wettbewerb import load_references

# Creating logger object
logger = logging.getLogger("main_log")

def oversampling(df):
    cols = df.columns
    y = df[['label']]
    X = df.drop("label", axis=1)
    oversample = SMOTE()

    X, y = oversample.fit_resample(X, y)

    df = pd.concat([X, y], axis=1)
    assert all(df.columns == cols)

    return df

def process_training_set() -> pd.DataFrame:

    ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name 
    features = []
    labels = []

    for idx, ecg_lead in enumerate(ecg_leads):
        logger.info(f"Processing {ecg_names[idx]}...")
        try:
            feature = pipeline(ecg_lead, fs)
            features.append(feature)
            
            label = ecg_labels[idx]
            labels.append(label)
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
        
        

    df = pd.DataFrame(features, columns=feature_names)
    s = pd.Series(labels, name="label")
    df = pd.concat([df, s], axis=1)

    df = oversampling(df)

    return df

def train_multilabel(df: pd.DataFrame) -> None:

    model_name = 'international_CO1.pkl'

    y_train = df[['label']]
    X_train = df.drop("label", axis=1)

    model = LGBMClassifier(
        metric="multi_logloss",
        num_leaves=32
    )
    
    train(X_train, y_train, model, model_name)

    logger.info(f"Saved model {model_name}")

def train_binary(df: pd.DataFrame) -> None:
    model_name = 'international_CO1_binary.pkl'

    df_b = df[(df['label'] != '~') & (df['label'] != 'O')]
    y_train = df_b[['label']]
    X_train = df_b.drop("label", axis=1)

    model = LGBMClassifier(
        metric="binary_logloss",
        num_leaves=32
    )

    train(X_train, y_train, model, model_name)
    logger.info(f"Saved model {model_name}")
    

def train(X_train, y_train, model, model_name):
    model.fit(X_train, y_train)

    joblib.dump(model, "models/" + model_name)


def main():
    global logger
    # Create and configure logger
    logging.basicConfig(filename="logfile.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    if not (pathlib.Path(__file__).parent / "features.csv").exists():
        df = process_training_set()
        df.to_csv("features.csv", index=False)
    else:
        df = pd.read_csv("features.csv")

    train_binary(df)
    train_multilabel(df)


if __name__ == "__main__":
    main()
