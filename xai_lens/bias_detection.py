import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class BiasDetector:
    def __init__(self, model, data, target, sensitive_features):
        self.model = model
        self.data = data

        if isinstance(target, str):
            if target in data.columns:
                self.target = data[target].values
            else:
                raise ValueError(f"Target column '{target}' not found in data")
        else:
            self.target = target

        self.sensitive_features = sensitive_features
        self.__validate_sensitive__features()

    def __validate_sensitive__features(self):
        if not all(feature in self.data.columns for feature in self.sensitive_features):
            missing = [f for f in self.sensitive_features if f not in self.data.columns]
            raise ValueError(f"Sensitive features {missing} not found in data")

        