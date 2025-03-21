import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class BiasDetector:
    """
    A class for detecting bias in machine learning models.
    Provides methods to analyze fairness across different demographic groups.
    """

    def __init__(self, model, data, target, sensitive_features):
        """
        Initialize the BiasDetector with a model, dataset, and sensitive attributes.

        Parameters:
        - model: Trained machine learning model
        - data: Feature dataset (pandas DataFrame)
        - target: Target variable name or array
        - sensitive_features: List of column names representing sensitive attributes
        """
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

    def calculate_disparate_impact(self, sensitive_feature, threshold=0.8):
        """
        Calculate disparate impact for a sensitive feature.

        A score < threshold indicates potential disparate impact.

        Parameters:
        - sensitive_feature: Name of the sensitive feature to analyze
        - threshold: Threshold for determining disparate impact (default: 0.8)

        Returns:
        - Dictionary with disparate impact scores and interpretation
        """
        if sensitive_feature not in self.sensitive_features:
            raise ValueError(f"'{sensitive_feature}' is not in the list of sensitive features")

        # TODO: make the below readable
        # Get predictions
        predictions = self.model.predict(self.data.drop(
            columns=[sensitive_feature] if isinstance(self.target, np.ndarray) else [sensitive_feature,
                                                                                     self.target.name]))
        # Calculate positive prediction rates for each group
        feature_values = self.data[sensitive_feature].unique()
        positive_rates = {}

        for value in feature_values:
            group_mask = self.data[sensitive_feature] == value
            group_predictions = predictions[group_mask]
            positive_rate = np.mean(group_predictions == 1) if 1 in group_predictions else 0
            positive_rates[value] = positive_rate

        # Find reference group (highest positive prediction rate)
        reference_group = max(positive_rates.items(), key=lambda x: x[1])

        impact_scores = {}

        for group, rate in positive_rates.items():
            if group == reference_group[0] or reference_group[1] == 0:
                impact_scores[group] = 1.0
            else:
                impact_scores[group] = rate / reference_group[1]

        min_score = min(impact_scores.values())
        assessment = "Potential disparate impact detected" if min_score < threshold else "No disparate impact detected"

        return {
            "scores": impact_scores,
            "reference_group": reference_group[0],
            "min_score": min_score,
            "threshold": threshold,
            "assessment": assessment
        }
