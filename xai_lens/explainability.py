import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

class ExplainabilityAnalyzer:
    """
    A core class for model explainability analysis.
    Supports SHAP explanations for various ML models.
    """


    def __init__(self, model, data):
        """
        Initialize the ExplainabilityAnalyzer with a model and dataset.
        Parameters:
        - model: Trained machine learning model (supports sklearn, XGBoost, etc.)
        - data: Feature dataset (numpy array or pandas dataframe)
        """

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
        """
        Plot a SHAP summary plot to visualize feature importance.
        """
        if self.explanations is None:
            print("No SHAP values computed yet. Must call generate_shap_values() first.")
            return

        print("Plotting SHAP summary plot...")
        shap.summary_plot(self.explanations.explanations, self.data)


    def get_top_features(self, n=3):
        """
        Return the top N most important features based on mean SHAP values.

        Parameters:
        - n: Number of top features to return (default: 5)

        Returns:
        - A list of top N feature names (or indices if unnamed)
        """
        if self.explanations is None:
            print("No SHAP values computed yet. Must call generate_shap_values() first.")
            return []

        mean_abs_shap_values = np.abs(self.explanations.values.mean(axis=0))
        top_features = np.argsort(mean_abs_shap_values)[::-1][:n]
        return top_features.tolist()