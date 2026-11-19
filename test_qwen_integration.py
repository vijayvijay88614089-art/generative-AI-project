import unittest
from unittest.mock import patch

from modules.analyst import analyze
from modules.code_generator import generate_expression
from modules.code_generator import local_expression
from modules.metadata import build_metadata
from tests.helpers import demo_frame


class QwenIntegrationTests(unittest.TestCase):
    def test_does_not_guess_metric_when_multiple_numeric_columns_exist(self):
        frame = demo_frame()
        self.assertIsNone(local_expression("What is the average?", frame))
        with patch("modules.transformer_inference.generate_local") as generate:
            actual, _, _, source = generate_expression(
                "What is the average?", frame, build_metadata(frame), local_model_id="qwen2.5-coder:1.5b"
            )
        self.assertEqual(actual, "")
        self.assertEqual(source, "Clarification required")
        generate.assert_not_called()

    def test_column_matching_uses_word_boundaries(self):
        frame = demo_frame()
        # A mention of "valid" must not accidentally select a column named "id".
        frame["id"] = range(len(frame))
        self.assertIsNone(local_expression("What is the total valid?", frame))

    def test_supported_question_uses_exact_local_plan_before_model(self):
        frame = demo_frame()
        question = "Calculate the median revenue grouped by region"
        expression = "df.groupby('region', dropna=False)['revenue'].median().to_dict()"
        with patch("modules.transformer_inference.generate_local", return_value=expression) as generate:
            actual, _, _, source = generate_expression(
                question, frame, build_metadata(frame), local_model_id="Qwen/Qwen2.5-Coder-7B-Instruct"
            )
        self.assertEqual(actual, expression)
        self.assertEqual(source, "Local rule-based planner")
        generate.assert_not_called()

    def test_model_load_failure_returns_an_app_result(self):
        with patch("modules.transformer_inference.generate_local", side_effect=OSError("weights unavailable")):
            result = analyze(
                "What is the latest sentiment across competitors?",
                demo_frame(),
                local_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            )
        self.assertFalse(result["ok"])
        self.assertIn("weights unavailable", result["error"])
        self.assertEqual(result["source"], "Model unavailable")


if __name__ == "__main__":
    unittest.main()
