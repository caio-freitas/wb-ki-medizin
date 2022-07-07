import joblib
import numpy as np
import pandas as pd
from preprocessing import pipeline
from typing import List, Tuple

def predict_labels(
    ecg_leads : List[np.ndarray], 
    fs : float, 
    ecg_names : List[str], 
    model_name : str='international_CO1',
    is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier  
    predictions = []
    data = []

    # process data
    for idx, ecg_lead in enumerate(ecg_leads):
        print(f"Processing {ecg_names[idx]}")
        processed_data = pipeline(ecg_signal=ecg_lead, sampling_freq=fs)
        data.append(processed_data)

        if ((idx+1) % 100)==0:
            print(str(idx+1) + "\t Dateien wurden verarbeitet.")

    X_test = np.array(data)


    # load model
    if len(model_name.split(".")) > 1:
        model_name = model_name[0]

    if is_binary_classifier:
        # loads binary model
        model = joblib.load("models/" + model_name + '_binary.pkl')
    else:
        # Loads multilabel model and file to be predicted
        model = joblib.load("models/" + model_name + '.pkl')
        

    # predict data
    y_test = model.predict(X_test)
    predictions = list(zip(ecg_names, y_test))
            
            
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!

