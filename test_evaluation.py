import unittest
from modules.evaluation import DEMO_CASES, equal
from modules.analyst import analyze
from tests.helpers import demo_frame


class EvaluationTests(unittest.TestCase):
    def test_fixed_gold_values_match_demo_frame(self):
        frame = demo_frame()
        for case in DEMO_CASES:
            with self.subTest(question=case["question"]):
                result = analyze(case["question"], frame)
                self.assertTrue(result["ok"], result.get("error"))
                self.assertEqual(result["answer_type"], case["expected_type"])
                self.assertTrue(equal(result["value"], case["expected"]), (result["value"], case["expected"]))

    def test_agent_default_comparison_is_supported_locally(self):
        from modules.ai_agent import run
        outcome = run("Compare revenue by region and show a chart", demo_frame())
        result = outcome["result"]
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(result["value"]["East"], 1266.0)
        statuses = {action["tool"]: action["status"] for action in outcome["actions"]}
        self.assertEqual(statuses["grouping and aggregation"], "completed")
        self.assertEqual(statuses["chart generation"], "completed")

    def test_agent_retrieves_document_evidence_when_requested(self):
        from modules.ai_agent import run
        from modules.rag_pipeline import Chunk

        evidence = [Chunk("policy.txt", None, "Refunds are allowed within thirty days of purchase.")]
        outcome = run("According to the policy document, when are refunds allowed?", demo_frame(), evidence)
        self.assertEqual(outcome["actions"][0]["tool"], "document retrieval")
        self.assertEqual(outcome["actions"][0]["status"], "completed")
        self.assertEqual(outcome["evidence"][0]["chunk"].source, "policy.txt")

    def test_unsupported_question_returns_guidance_without_exception(self):
        result = analyze("What is the latest sentiment across competitors?", demo_frame())
        self.assertFalse(result["ok"])
        self.assertIn("outside the local planner", result["error"])


if __name__ == "__main__": unittest.main()
