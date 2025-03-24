

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
        