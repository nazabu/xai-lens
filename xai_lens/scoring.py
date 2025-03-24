

class InterpretabilityScorer:
    def __init__(self, model, data=None):
        self.model = model
        self.data = data
        self.model_type  = None # model type function

