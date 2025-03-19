import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xai_lens.explainability import ExplainabilityAnalyzer

# your data
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# train model
model = RandomForestClassifier()
model.fit(X, y)

# Initialize the explainability analyzer
analyzer = ExplainabilityAnalyzer(model, X)

# Generate and visualize SHAP values
analyzer.generate_shap_values()
analyzer.plot_shap_summary()

# Get top 3 most important features
top_features = analyzer.get_top_features(n=3)
print(f"Top 3 most important features: {top_features}")