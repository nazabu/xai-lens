# XAI-Lens

A universal explainability analysis toolkit for machine learning models. XAI-Lens provides straightforward tools for generating SHAP/LIME explanations, detecting bias across sensitive attributes, scoring model interpretability, and compiling structured reports.

## Features

- **Explainability Analysis** -- SHAP and LIME explanations for any scikit-learn-compatible model
- **Bias Detection** -- Disparate impact and equal opportunity metrics across demographic groups
- **Interpretability Scoring** -- Automatic scoring of how interpretable a model is (0-1 scale)
- **Report Generation** -- Compile all results into a single report (text summary or JSON export)

## Installation

```bash
pip install xai-lens
```

Or install from source:

```bash
git clone https://github.com/nazabu/xai-lens.git
cd xai-lens
pip install -e .
```

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xai_lens import ExplainabilityAnalyzer, InterpretabilityScorer

X, y = make_classification(n_samples=200, n_features=5, random_state=42)
model = RandomForestClassifier()
model.fit(X, y)

# SHAP explainability
analyzer = ExplainabilityAnalyzer(model, X)
analyzer.generate_shap_values()
print(analyzer.get_top_features(n=3))

# Interpretability score
scorer = InterpretabilityScorer(model, X)
print(scorer.calculate_score())
```

## Usage

### Explainability Analysis

```python
from xai_lens import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model, X)

# SHAP values
analyzer.generate_shap_values()
analyzer.plot_shap_summary()
top_features = analyzer.get_top_features(n=5)

# LIME explanation for a single instance
explanation = analyzer.generate_lime_explanation(instance_index=0)
```

### Bias Detection

```python
import pandas as pd
from xai_lens import BiasDetector

df = pd.DataFrame(X, columns=feature_names)
df["group"] = group_labels

detector = BiasDetector(
    model=model,
    data=df,
    target=y,
    sensitive_features=["group"],
)

# Single metric
di = detector.calculate_disparate_impact("group")
eo = detector.calculate_equal_opportunity("group")

# Full report across all sensitive features
report = detector.generate_bias_report()
```

### Interpretability Scoring

```python
from xai_lens import InterpretabilityScorer

scorer = InterpretabilityScorer(model, X)
result = scorer.calculate_score()
# {
#   "model_type": "tree",
#   "complexity": 0.5,
#   "feature_importance_availability": 1.0,
#   "n_features_score": 1.0,
#   "overall_score": 0.8,
#   "rating": "Highly Interpretable"
# }
```

### Report Generation

```python
from xai_lens import ReportGenerator

report = ReportGenerator(model_name="My Model")
report.add_explainability_results(top_features, feature_names=feature_names)
report.add_bias_report(detector.generate_bias_report())
report.add_interpretability_score(scorer.calculate_score())

# Print a text summary
print(report.summary())

# Export to JSON
report.to_json("report.json")
```

## Project Structure

```
xai-lens/
├── xai_lens/
│   ├── __init__.py              # Package exports
│   ├── explainability.py        # SHAP & LIME analysis
│   ├── bias_detection.py        # Fairness metrics
│   ├── scoring.py               # Interpretability scoring
│   ├── reports.py               # Report compilation
│   └── utils.py                 # Helpers (normalize, load_model)
├── tests/                       # Unit tests
├── examples/
│   └── demo.py                  # End-to-end example
├── setup.py
├── requirements.txt
└── pyproject.toml
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Dependencies

- numpy, pandas, scikit-learn
- shap, lime
- matplotlib, joblib

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to GitHub (`git push origin feature-name`)
5. Open a Pull Request

Please run `pytest tests/` before submitting.

## License

[MIT](LICENSE)
