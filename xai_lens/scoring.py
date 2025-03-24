

class InterpretabilityScorer:
    """
    A class for scoring model interpretability based on various metrics.
    Provides methods to calculate and aggregate interpretability scores.
    """
    def __init__(self, model, data=None):
        self.model = model
        self.data = data
        self.model_type  = self._determine_model_type()

    def _determine_model_type(self):
        pass