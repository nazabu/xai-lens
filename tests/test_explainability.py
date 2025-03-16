import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Generate a sample dataset and trained model for testing."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, X

