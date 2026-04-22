import numpy as np


class InterpretabilityScorer:
    """
    A class for scoring model interpretability based on various metrics.
    Provides methods to calculate and aggregate interpretability scores.

    The overall score ranges from 0.0 to 1.0 where higher values indicate
    greater interpretability.
    """

    def __init__(self, model, data=None):
        """
        Initialize the InterpretabilityScorer with a model and optional dataset.

        Parameters:
        - model: Trained machine learning model
        - data: Optional feature dataset for calculating complexity metrics
        """
        self.model = model
        self.data = data
        self.model_type = self._determine_model_type()

    def _determine_model_type(self):
        """Classify the model into a known category for scoring."""
        model_name = type(self.model).__name__.lower()

        if "linear" in model_name or "logistic" in model_name:
            return "linear"
        elif "tree" in model_name or "forest" in model_name or "boost" in model_name:
            return "tree"
        elif "svm" in model_name or "svc" in model_name or "svr" in model_name:
            return "svm"
        elif "mlp" in model_name or "neural" in model_name:
            return "neural"
        else:
            return "unknown"

    def calculate_model_complexity(self):
        """
        Estimate model complexity on a 0.0-1.0 scale.
        Lower values = simpler model = more interpretable.

        Returns:
        - float: complexity score between 0.0 and 1.0
        """
        if self.model_type == "linear":
            if hasattr(self.model, "coef_"):
                n_features = np.count_nonzero(self.model.coef_)
                return min(0.2 + (n_features / 1000), 0.3)
            return 0.2

        elif self.model_type == "tree":
            if hasattr(self.model, "get_depth"):
                # Single decision tree
                depth = self.model.get_depth()
                return min(0.3 + (depth / 20), 0.7)
            elif hasattr(self.model, "estimators_"):
                # Ensemble (RandomForest, GradientBoosting, etc.)
                n_estimators = len(self.model.estimators_)
                return min(0.4 + (n_estimators / 500), 0.8)
            return 0.5

        elif self.model_type == "svm":
            if hasattr(self.model, "kernel") and self.model.kernel == "linear":
                return 0.4
            return 0.7

        elif self.model_type == "neural":
            return 0.9

        else:
            return 0.8

    def calculate_feature_importance_availability(self):
        """
        Check whether the model exposes built-in feature importance.
        Models with transparent feature importance are more interpretable.

        Returns:
        - float: 1.0 if feature_importances_ available, 0.8 if coef_
          available, 0.0 otherwise
        """
        if hasattr(self.model, "feature_importances_"):
            return 1.0
        elif hasattr(self.model, "coef_"):
            return 0.8
        return 0.0

    def calculate_n_features_score(self):
        """
        Score based on the number of input features. Fewer features
        generally means a more interpretable model.

        Returns:
        - float: score between 0.0 and 1.0 (higher = more interpretable)
        """
        if self.data is None:
            return 0.5

        n_features = self.data.shape[1] if hasattr(self.data, "shape") else 0
        if n_features <= 5:
            return 1.0
        elif n_features <= 15:
            return 0.8
        elif n_features <= 50:
            return 0.5
        else:
            return 0.2

    def calculate_score(self):
        """
        Calculate an overall interpretability score by combining sub-metrics.

        Weights:
        - Model complexity: 40%
        - Feature importance availability: 30%
        - Number of features: 30%

        Returns:
        - Dictionary with individual metric scores and the overall score
        """
        complexity = self.calculate_model_complexity()
        importance_avail = self.calculate_feature_importance_availability()
        n_features_score = self.calculate_n_features_score()

        # Invert complexity so that lower complexity = higher interpretability
        interpretability_from_complexity = 1.0 - complexity

        overall = (
            0.4 * interpretability_from_complexity
            + 0.3 * importance_avail
            + 0.3 * n_features_score
        )

        return {
            "model_type": self.model_type,
            "complexity": round(complexity, 3),
            "feature_importance_availability": round(importance_avail, 3),
            "n_features_score": round(n_features_score, 3),
            "overall_score": round(overall, 3),
            "rating": self._score_to_rating(overall),
        }

    @staticmethod
    def _score_to_rating(score):
        """Convert a numeric score to a human-readable rating."""
        if score >= 0.8:
            return "Highly Interpretable"
        elif score >= 0.6:
            return "Moderately Interpretable"
        elif score >= 0.4:
            return "Low Interpretability"
        else:
            return "Very Low Interpretability"
