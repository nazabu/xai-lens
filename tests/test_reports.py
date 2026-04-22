import json
import os
import tempfile
import pytest
from xai_lens.reports import ReportGenerator


@pytest.fixture
def generator():
    return ReportGenerator(model_name="TestModel")


def test_compile_empty(generator):
    report = generator.compile()
    assert report["model_name"] == "TestModel"
    assert "generated_at" in report
    assert report["sections"] == {}


def test_add_interpretability_score(generator):
    score_result = {
        "model_type": "tree",
        "complexity": 0.5,
        "feature_importance_availability": 1.0,
        "n_features_score": 0.8,
        "overall_score": 0.74,
        "rating": "Moderately Interpretable",
    }
    generator.add_interpretability_score(score_result)
    report = generator.compile()
    assert "interpretability" in report["sections"]
    assert report["sections"]["interpretability"]["overall_score"] == 0.74


def test_add_bias_report(generator):
    bias_result = {
        "gender": {
            "disparate_impact": {
                "scores": {"M": 1.0, "F": 0.85},
                "reference_group": "M",
                "min_score": 0.85,
                "threshold": 0.8,
                "assessment": "No disparate impact detected",
            },
            "equal_opportunity": {
                "true_positive_rates": {"M": 0.9, "F": 0.88},
                "max_difference": 0.02,
                "assessment": "No significant bias detected",
            },
        }
    }
    generator.add_bias_report(bias_result)
    report = generator.compile()
    assert "bias" in report["sections"]
    assert "gender" in report["sections"]["bias"]


def test_add_explainability_results(generator):
    generator.add_explainability_results([0, 2, 4], feature_names=["a", "b", "c", "d", "e"])
    report = generator.compile()
    section = report["sections"]["explainability"]
    assert section["top_feature_names"] == ["a", "c", "e"]


def test_add_explainability_results_no_names(generator):
    generator.add_explainability_results([0, 1, 3])
    report = generator.compile()
    section = report["sections"]["explainability"]
    assert section["top_feature_names"] == ["Feature 0", "Feature 1", "Feature 3"]


def test_to_json(generator):
    generator.add_interpretability_score({"overall_score": 0.7, "rating": "Moderately Interpretable"})
    json_str = generator.to_json()
    parsed = json.loads(json_str)
    assert parsed["model_name"] == "TestModel"


def test_to_json_file(generator):
    generator.add_interpretability_score({"overall_score": 0.7, "rating": "OK"})
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        generator.to_json(path=path)
        with open(path) as f:
            parsed = json.loads(f.read())
        assert parsed["model_name"] == "TestModel"
    finally:
        os.unlink(path)


def test_summary(generator):
    generator.add_interpretability_score({
        "model_type": "tree",
        "complexity": 0.5,
        "overall_score": 0.74,
        "rating": "Moderately Interpretable",
    })
    generator.add_explainability_results([0, 1], feature_names=["age", "income"])
    generator.add_bias_report({
        "gender": {
            "disparate_impact": {"min_score": 0.85, "assessment": "No disparate impact detected"},
            "equal_opportunity": {"max_difference": 0.02, "assessment": "No significant bias detected"},
        }
    })
    text = generator.summary()
    assert "TestModel" in text
    assert "0.74" in text
    assert "age" in text
    assert "gender" in text
