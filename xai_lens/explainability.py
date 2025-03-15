import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

class ExplainabilityAnalyzer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.explainer = None
        self.explanations = None

    