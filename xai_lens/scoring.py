import numpy as np

class InterpretabilityScorer:
    """
    A class for scoring model interpretability based on various metrics.
    Provides methods to calculate and aggregate interpretability scores.
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
        self.model_type  = self._determine_model_type()

    def _determine_model_type(self):
        model_name = str(type(self.model).__name__).lower()

        if "linear" in model_name or "logistic" in model_name:
            return "linear"
        elif "tree" in model_name or "forest" in model_name or "boost" in model_name:
            return "tree"
        elif "svm" in model_name:
            return "svm"
        else:
            return "unknown"

    def calculate_model_complexity(self):

        if self.model_type == "linear":
            # Generally, linear models are highly interpretable
            try:
                # count non-zero coefficients
                if hasattr(self.model, 'coef_'):
                    n_features = np.count_nonzero(self.model.coef_)
                else:
                    n_features = 0

                # More features = slightly lower interpretability, but still high
                return min(0.2 + (n_features/1000), 0.3)
            except:
                return 0.2

        elif self.model_type == "tree":
            try:
                if hasattr(self.model, 'get_depth'):
                    depth = self.model.get_depth()
                    return min(0.3 + (depth / 20), 0.7)
                elif hasattr(self.model, 'estimators_'):
                    n_estimators = self.model.get_n_estimators()
                    return min(0.4 + (n_estimators / 500), 0.8)
                else:
                    return 0.5
            except:
                return 0.5

        elif self.model_type == "svm":
            try:
                if hasattr(self.model, 'kernel') and self.model.kernel == "linear":
                    return 0.6
                else:
                    return 0.8
            except:
                return 0.7
