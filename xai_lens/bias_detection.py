import numpy as np
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
        - target: Target variable name (str) or array-like of target values
        - sensitive_features: List of column names representing sensitive attributes
        """
        self.model = model
        self.data = data

        if isinstance(target, str):
            if target in data.columns:
                self.target_column = target
                self.target = data[target].values
            else:
                raise ValueError(f"Target column '{target}' not found in data")
        else:
            self.target_column = None
            self.target = np.asarray(target)

        self.sensitive_features = sensitive_features
        self._validate_sensitive_features()

    def _validate_sensitive_features(self):
        missing = [f for f in self.sensitive_features if f not in self.data.columns]
        if missing:
            raise ValueError(f"Sensitive features {missing} not found in data")

    def _get_feature_columns(self, sensitive_feature):
        """Return the data columns to use for prediction, excluding sensitive and target columns."""
        drop_cols = [sensitive_feature]
        if self.target_column and self.target_column in self.data.columns:
            drop_cols.append(self.target_column)
        return self.data.drop(columns=drop_cols)

    def calculate_disparate_impact(self, sensitive_feature, threshold=0.8):
        """
        Calculate disparate impact for a sensitive feature.

        Disparate impact compares the positive prediction rate of each group
        against the group with the highest rate. A ratio below the threshold
        (commonly 0.8, the "four-fifths rule") indicates potential bias.

        Parameters:
        - sensitive_feature: Name of the sensitive feature to analyze
        - threshold: Threshold for determining disparate impact (default: 0.8)

        Returns:
        - Dictionary with disparate impact scores and interpretation
        """
        if sensitive_feature not in self.sensitive_features:
            raise ValueError(f"'{sensitive_feature}' is not in the list of sensitive features")

        features = self._get_feature_columns(sensitive_feature)
        predictions = self.model.predict(features)

        # Calculate positive prediction rate for each group
        feature_values = self.data[sensitive_feature].unique()
        positive_rates = {}
        for value in feature_values:
            group_mask = self.data[sensitive_feature] == value
            group_preds = predictions[group_mask]
            positive_rates[value] = np.mean(group_preds == 1) if len(group_preds) > 0 else 0.0

        # Reference group = highest positive prediction rate
        reference_group = max(positive_rates, key=positive_rates.get)
        reference_rate = positive_rates[reference_group]

        # Calculate disparate impact ratios
        impact_scores = {}
        for group, rate in positive_rates.items():
            if group == reference_group or reference_rate == 0:
                impact_scores[group] = 1.0
            else:
                impact_scores[group] = rate / reference_rate

        min_score = min(impact_scores.values())
        assessment = (
            "Potential disparate impact detected"
            if min_score < threshold
            else "No disparate impact detected"
        )

        return {
            "scores": impact_scores,
            "reference_group": reference_group,
            "min_score": min_score,
            "threshold": threshold,
            "assessment": assessment,
        }

    def calculate_equal_opportunity(self, sensitive_feature):
        """
        Calculate equal opportunity difference for a sensitive feature.

        Equal opportunity measures the difference in true positive rates
        across groups. A large difference indicates the model is better
        at correctly identifying positives for some groups than others.

        Parameters:
        - sensitive_feature: Name of the sensitive feature to analyze

        Returns:
        - Dictionary with true positive rates per group and max difference
        """
        if sensitive_feature not in self.sensitive_features:
            raise ValueError(f"'{sensitive_feature}' is not in the list of sensitive features")

        features = self._get_feature_columns(sensitive_feature)
        predictions = self.model.predict(features)

        feature_values = self.data[sensitive_feature].unique()
        true_positive_rates = {}

        for value in feature_values:
            group_mask = self.data[sensitive_feature] == value
            group_preds = predictions[group_mask]
            group_target = self.target[group_mask]

            tn, fp, fn, tp = confusion_matrix(
                group_target, group_preds, labels=[0, 1]
            ).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            true_positive_rates[value] = tpr

        max_diff = max(true_positive_rates.values()) - min(true_positive_rates.values())

        return {
            "true_positive_rates": true_positive_rates,
            "max_difference": max_diff,
            "assessment": (
                "Potential bias detected"
                if max_diff > 0.1
                else "No significant bias detected"
            ),
        }

    def generate_bias_report(self):
        """
        Generate a comprehensive bias report for all sensitive features.

        Returns:
        - Dictionary with disparate impact and equal opportunity metrics
          for each sensitive feature
        """
        report = {}
        for feature in self.sensitive_features:
            report[feature] = {
                "disparate_impact": self.calculate_disparate_impact(feature),
                "equal_opportunity": self.calculate_equal_opportunity(feature),
            }
        return report
