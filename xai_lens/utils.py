import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def normalize(data):
    """
    Normalize a dataset using standard scaling (zero mean, unit variance).

    Parameters:
    - data: numpy array or pandas dataframe

    Returns:
    - Scaled dataset (numpy array)
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def load_model(path):
    """
    Load a saved machine learning model from a file.

    Parameters:
    - path: Path to the saved model file (.pkl)

    Returns:
    - Loaded model object
    """
    try:
        model = joblib.load(path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model {e}")
        return None