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

    def generate_shap_values(self):
        print("Generating shap values...")
        try:
            self.explainer = shap.KernelExplainer(model=self.model, data=self.data)
            self.explainations = self.explainer(self.data)
        except Exception as e:
            print(f"Error computing SHAP values: {e}")


    def plot_shap_summary(self):
        if self.explanations is None:
            print("No SHAP values computed yet. Must call generate_shap_values() first.")
            return

        print("Plotting SHAP summary plot...")
        shap.summary_plot(self.explanations.explanations, self.data)

    