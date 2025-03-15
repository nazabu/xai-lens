import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def normalize(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def load_model(path):
    try:
        model = joblib.load(path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model {e}")
        return None