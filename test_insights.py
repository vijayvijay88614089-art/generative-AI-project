import unittest

import pandas as pd

from modules.insights import build_html_report, generate_insights


class InsightTests(unittest.TestCase):
    def test_finds_missing_values_duplicates_outliers_and_correlation(self):
        frame = pd.DataFrame({
            "revenue": [10, 20, 30, 40, 1000, None, 10],
            "revenue_copy": [10, 20, 30, 40, 1000, None, 10],
            "region": ["East", "East", "East", "West", "East", "West", "East"],
        })
        result = generate_insights(frame)
        text = " ".join(item["insight"] + " " + item["evidence"] for item in result["findings"])
        self.assertIn("Missing values in revenue", text)
        self.assertIn("Potential outliers in revenue", text)
        self.assertIn("Strong correlation", text)
        self.assertIn("1 duplicate rows", text)
        self.assertEqual(result["rows"], 7)
        self.assertEqual(result["columns"], 3)

    def test_empty_dataset_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "at least one row"):
            generate_insights(pd.DataFrame())

    def test_html_report_escapes_uploaded_column_names(self):
        result = generate_insights(pd.DataFrame({"<script>alert(1)</script>": [1, 2]}))
        report = build_html_report(result, "<img src=x onerror=alert(1)>")
        self.assertNotIn("<script>alert(1)</script>", report)
        self.assertNotIn("<img src=x", report)
        self.assertIn("&lt;script&gt;", report)


if __name__ == "__main__":
    unittest.main()
