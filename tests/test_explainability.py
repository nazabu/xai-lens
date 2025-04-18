import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xai_lens.explainability import ExplainabilityAnalyzer

"""Generate a sample dataset and trained model for testing."""
@pytest.fixture
def sample_data():
    model = RandomForestClassifier()
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model.fit(X, y)
    return model, X

"""Test if SHAP values are generated successfully."""
def test_generate_shap_values(sample_data):
    model, X = sample_data
    analyzer = ExplainabilityAnalyzer(model, X)

    analyzer.generate_shap_values()

    assert analyzer.explanations is not None, "SHAP values should not be None."


"""Test if top feature indices are returned correctly."""
def test_get_top_features(sample_data):
    model, X = sample_data
    analyzer = ExplainabilityAnalyzer(model, X)

    analyzer.generate_shap_values()
    top_features = analyzer.get_top_features(n=3)

    assert isinstance(top_features, list), "Output should be a list."
    assert len(top_features) == 3, "Should return exactly 3 features."
    assert all(isinstance(i, int) for i in top_features), "Feature indices should be integers."