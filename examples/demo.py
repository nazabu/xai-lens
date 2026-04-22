"""
XAI-Lens Demo
==============
Demonstrates how to use all major components of the library:
  1. Explainability analysis (SHAP + LIME)
  2. Bias detection
  3. Interpretability scoring
  4. Report generation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from xai_lens import (
    ExplainabilityAnalyzer,
    BiasDetector,
    InterpretabilityScorer,
    ReportGenerator,
)

# ---------------------------------------------------------------------------
# 1. Prepare a synthetic dataset
# ---------------------------------------------------------------------------
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
feature_names = ["age", "income", "score", "history", "balance"]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# ---------------------------------------------------------------------------
# 2. Explainability Analysis
# ---------------------------------------------------------------------------
print("--- Explainability Analysis ---")
analyzer = ExplainabilityAnalyzer(model, X)

analyzer.generate_shap_values()
top = analyzer.get_top_features(n=3)
print(f"Top 3 features (by SHAP): {[feature_names[i] for i in top]}")

# Generate LIME explanation for a single instance
explanation = analyzer.generate_lime_explanation(instance_index=0)
if explanation:
    print(f"LIME explanation generated for instance 0")

# Uncomment to show the SHAP summary plot:
# analyzer.plot_shap_summary()

# ---------------------------------------------------------------------------
# 3. Bias Detection
# ---------------------------------------------------------------------------
print("\n--- Bias Detection ---")
df = pd.DataFrame(X, columns=feature_names)
df["group"] = np.random.choice(["A", "B"], size=len(X), replace=True)

detector = BiasDetector(model=model, data=df[feature_names + ["group"]],
                        target=y, sensitive_features=["group"])

bias_report = detector.generate_bias_report()
for feat, metrics in bias_report.items():
    di = metrics["disparate_impact"]
    eo = metrics["equal_opportunity"]
    print(f"  {feat}:")
    print(f"    Disparate Impact  — min score: {di['min_score']:.3f} ({di['assessment']})")
    print(f"    Equal Opportunity — max diff:  {eo['max_difference']:.3f} ({eo['assessment']})")

# ---------------------------------------------------------------------------
# 4. Interpretability Scoring
# ---------------------------------------------------------------------------
print("\n--- Interpretability Score ---")
scorer = InterpretabilityScorer(model, X)
score = scorer.calculate_score()
print(f"  Model type : {score['model_type']}")
print(f"  Complexity : {score['complexity']}")
print(f"  Overall    : {score['overall_score']} ({score['rating']})")

# ---------------------------------------------------------------------------
# 5. Compiled Report
# ---------------------------------------------------------------------------
print("\n--- Full Report ---")
report = ReportGenerator(model_name="Demo RandomForest")
report.add_explainability_results(top, feature_names=feature_names)
report.add_bias_report(bias_report)
report.add_interpretability_score(score)

print(report.summary())

# Optionally export to JSON:
# report.to_json("report.json")
