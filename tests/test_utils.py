import os
import tempfile
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xai_lens.utils import normalize, load_model


def test_normalize():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    result = normalize(data)
    assert result.shape == data.shape
    np.testing.assert_almost_equal(result.mean(axis=0), [0, 0])
    np.testing.assert_almost_equal(result.std(axis=0), [1, 1])


def test_normalize_single_column():
    data = np.array([[10], [20], [30]])
    result = normalize(data)
    assert result.shape == (3, 1)


def test_load_model_success():
    import joblib

    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        joblib.dump(model, path)
        loaded = load_model(path)
        assert loaded is not None
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    finally:
        os.unlink(path)


def test_load_model_bad_path():
    result = load_model("/nonexistent/path/model.pkl")
    assert result is None
