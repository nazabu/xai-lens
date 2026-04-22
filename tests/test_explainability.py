import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xai_lens.explainability import ExplainabilityAnalyzer


@pytest.fixture
def sample_data():
    """Generate a sample dataset and trained model for testing."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X


def test_generate_shap_values(sample_data):
    """Test if SHAP values are generated successfully."""
    model, X = sample_data
    analyzer = ExplainabilityAnalyzer(model, X)
    analyzer.generate_shap_values()
    assert analyzer.explanations is not None, "SHAP values should not be None."


def test_get_top_features(sample_data):
    """Test if top feature indices are returned correctly."""
    model, X = sample_data
    analyzer = ExplainabilityAnalyzer(model, X)
    analyzer.generate_shap_values()
    top_features = analyzer.get_top_features(n=3)

    assert isinstance(top_features, list), "Output should be a list."
    assert len(top_features) == 3, "Should return exactly 3 features."
    assert all(isinstance(i, int) for i in top_features), "Feature indices should be integers."


def test_get_top_features_before_shap(sample_data):
    """Test that get_top_features returns empty list when SHAP hasn't been run."""
    model, X = sample_data
    analyzer = ExplainabilityAnalyzer(model, X)
    result = analyzer.get_top_features(n=3)
    assert result == []


def test_generate_lime_explanation(sample_data):
    """Test that LIME explanation is generated successfully."""
    model, X = sample_data
    analyzer = ExplainabilityAnalyzer(model, X)
    explanation = analyzer.generate_lime_explanation(instance_index=0)
    assert explanation is not None, "LIME explanation should not be None."
