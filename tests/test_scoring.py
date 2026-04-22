import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from xai_lens.scoring import InterpretabilityScorer


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y


def test_linear_model_type(classification_data):
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    assert scorer.model_type == "linear"


def test_tree_model_type(classification_data):
    X, y = classification_data
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    assert scorer.model_type == "tree"


def test_svm_model_type(classification_data):
    X, y = classification_data
    model = SVC(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    assert scorer.model_type == "svm"


def test_complexity_linear(classification_data):
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    complexity = scorer.calculate_model_complexity()
    assert 0.0 <= complexity <= 0.3


def test_complexity_tree(classification_data):
    X, y = classification_data
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    complexity = scorer.calculate_model_complexity()
    assert 0.3 <= complexity <= 0.7


def test_complexity_ensemble(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    complexity = scorer.calculate_model_complexity()
    assert 0.4 <= complexity <= 0.8


def test_feature_importance_available(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    assert scorer.calculate_feature_importance_availability() == 1.0


def test_coef_importance(classification_data):
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    assert scorer.calculate_feature_importance_availability() == 0.8


def test_no_importance(classification_data):
    X, y = classification_data
    model = SVC(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    assert scorer.calculate_feature_importance_availability() == 0.0


def test_calculate_score_structure(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, X)
    result = scorer.calculate_score()

    assert "model_type" in result
    assert "complexity" in result
    assert "feature_importance_availability" in result
    assert "n_features_score" in result
    assert "overall_score" in result
    assert "rating" in result
    assert 0.0 <= result["overall_score"] <= 1.0


def test_n_features_score_no_data(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    scorer = InterpretabilityScorer(model, data=None)
    assert scorer.calculate_n_features_score() == 0.5


def test_score_rating_values():
    """Test that _score_to_rating covers all ranges."""
    assert InterpretabilityScorer._score_to_rating(0.9) == "Highly Interpretable"
    assert InterpretabilityScorer._score_to_rating(0.7) == "Moderately Interpretable"
    assert InterpretabilityScorer._score_to_rating(0.5) == "Low Interpretability"
    assert InterpretabilityScorer._score_to_rating(0.2) == "Very Low Interpretability"
