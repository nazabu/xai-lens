import json
from datetime import datetime


class ReportGenerator:
    """
    Generates structured explainability reports that combine results
    from the ExplainabilityAnalyzer, BiasDetector, and InterpretabilityScorer.
    """

    def __init__(self, model_name=None):
        """
        Initialize the ReportGenerator.

        Parameters:
        - model_name: Optional human-readable name for the model
        """
        self.model_name = model_name or "Unnamed Model"
        self._sections = {}

    def add_interpretability_score(self, scorer_result):
        """
        Add interpretability scoring results.

        Parameters:
        - scorer_result: Dictionary returned by InterpretabilityScorer.calculate_score()
        """
        self._sections["interpretability"] = scorer_result

    def add_bias_report(self, bias_result):
        """
        Add bias detection results.

        Parameters:
        - bias_result: Dictionary returned by BiasDetector.generate_bias_report()
        """
        self._sections["bias"] = bias_result

    def add_explainability_results(self, top_features, feature_names=None):
        """
        Add explainability analysis results.

        Parameters:
        - top_features: List of top feature indices from ExplainabilityAnalyzer.get_top_features()
        - feature_names: Optional list of feature names to map indices to names
        """
        if feature_names:
            named = [feature_names[i] for i in top_features if i < len(feature_names)]
        else:
            named = [f"Feature {i}" for i in top_features]

        self._sections["explainability"] = {
            "top_feature_indices": top_features,
            "top_feature_names": named,
        }

    def compile(self):
        """
        Compile all added sections into a structured report dictionary.

        Returns:
        - Dictionary containing the full report with metadata
        """
        return {
            "model_name": self.model_name,
            "generated_at": datetime.now().isoformat(),
            "sections": dict(self._sections),
        }

    def to_json(self, path=None, indent=2):
        """
        Export the report as a JSON string, optionally writing to a file.

        Parameters:
        - path: Optional file path to write the JSON to
        - indent: JSON indentation level (default: 2)

        Returns:
        - JSON string of the compiled report
        """
        report = self.compile()

        def _default(obj):
            """Handle numpy types that aren't JSON-serializable."""
            import numpy as np

            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json_str = json.dumps(report, indent=indent, default=_default)

        if path:
            with open(path, "w") as f:
                f.write(json_str)

        return json_str

    def summary(self):
        """
        Return a concise text summary of the report.

        Returns:
        - Multi-line string summarizing key findings
        """
        report = self.compile()
        lines = [
            f"=== XAI-Lens Report: {report['model_name']} ===",
            f"Generated: {report['generated_at']}",
            "",
        ]

        sections = report["sections"]

        if "interpretability" in sections:
            s = sections["interpretability"]
            lines.append(f"Interpretability Score: {s['overall_score']} ({s['rating']})")
            lines.append(f"  Model type: {s['model_type']}")
            lines.append(f"  Complexity: {s['complexity']}")
            lines.append("")

        if "explainability" in sections:
            s = sections["explainability"]
            lines.append(f"Top Features: {', '.join(s['top_feature_names'])}")
            lines.append("")

        if "bias" in sections:
            lines.append("Bias Analysis:")
            for feature, metrics in sections["bias"].items():
                di = metrics["disparate_impact"]
                eo = metrics["equal_opportunity"]
                lines.append(f"  {feature}:")
                lines.append(f"    Disparate Impact: {di['min_score']:.3f} — {di['assessment']}")
                lines.append(f"    Equal Opportunity Diff: {eo['max_difference']:.3f} — {eo['assessment']}")
            lines.append("")

        return "\n".join(lines)
