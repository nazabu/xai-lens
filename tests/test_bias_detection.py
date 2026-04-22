import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xai_lens.bias_detection import BiasDetector


@pytest.fixture
def bias_data():
    """Generate a dataset with a sensitive feature for bias testing."""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "group": np.random.choice([0, 1], size=n),
        "target": np.random.choice([0, 1], size=n),
    })

    # Train on only the non-sensitive features
    model = RandomForestClassifier(random_state=42)
    model.fit(data[["feature_1", "feature_2"]], data["target"])

    return model, data[["feature_1", "feature_2", "group"]], data["target"], ["group"]


def test_init_with_target_array(bias_data):
    """Test initialization with target as an array."""
    model, data, target, sensitive = bias_data
    detector = BiasDetector(model, data, target.values, sensitive)
    assert detector.target is not None


def test_init_missing_sensitive_feature(bias_data):
    """Test that missing sensitive features raise ValueError."""
    model, data, target, _ = bias_data
    with pytest.raises(ValueError, match="not found in data"):
        BiasDetector(model, data, target.values, ["nonexistent"])


def test_disparate_impact(bias_data):
    """Test disparate impact calculation returns expected structure."""
    model, data, target, sensitive = bias_data
    detector = BiasDetector(model, data, target.values, sensitive)
    result = detector.calculate_disparate_impact("group")

    assert "scores" in result
    assert "reference_group" in result
    assert "min_score" in result
    assert "threshold" in result
    assert "assessment" in result
    assert 0.0 <= result["min_score"] <= 1.0


def test_equal_opportunity(bias_data):
    """Test equal opportunity calculation returns expected structure."""
    model, data, target, sensitive = bias_data
    detector = BiasDetector(model, data, target.values, sensitive)
    result = detector.calculate_equal_opportunity("group")

    assert "true_positive_rates" in result
    assert "max_difference" in result
    assert "assessment" in result
    assert result["max_difference"] >= 0.0


def test_generate_bias_report(bias_data):
    """Test full bias report generation."""
    model, data, target, sensitive = bias_data
    detector = BiasDetector(model, data, target.values, sensitive)
    report = detector.generate_bias_report()

    assert "group" in report
    assert "disparate_impact" in report["group"]
    assert "equal_opportunity" in report["group"]


def test_invalid_sensitive_feature_in_analysis(bias_data):
    """Test that analyzing a non-registered sensitive feature raises ValueError."""
    model, data, target, sensitive = bias_data
    detector = BiasDetector(model, data, target.values, sensitive)
    with pytest.raises(ValueError, match="is not in the list"):
        detector.calculate_disparate_impact("feature_1")
