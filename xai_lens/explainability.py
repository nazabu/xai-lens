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


    def get_top_features(self, n=3):
        if self.explanations is None:
            print("No SHAP values computed yet. Must call generate_shap_values() first.")
            return []

        mean_abs_shap_values = np.abs(self.explanations.values.mean(axis=0))
        top_features = np.argsort(mean_abs_shap_values)[::-1][:n]
        return top_features.tolist()