import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xai_lens.explainability import ExplainabilityAnalyzer

@pytest.fixture
def sample_data():
    """Generate a sample dataset and trained model for testing."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, X


def test_generate_shap_values(sample_data):
    """Test if SHAP values are generated successfully."""
    model, X = sample_data
    analyzer = ExplainabilityAnalyzer(model, X)

    analyzer.generate_shap_values()

    assert analyzer.explanations is not None, "SHAP values should not be None."