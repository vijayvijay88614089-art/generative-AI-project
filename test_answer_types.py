import unittest
from modules.answer_classifier import classify


class AnswerTypeTests(unittest.TestCase):
    def test_types(self):
        samples = {
            "Is total revenue greater than 50?":"boolean",
            "Which region has highest revenue?":"category",
            "Show top 3 products by revenue":"list[category]",
            "What is average unit_price?":"number",
            "How many rows are in this dataset?":"number",
        }
        for question, answer_type in samples.items():
            with self.subTest(question=question): self.assertEqual(classify(question), answer_type)


if __name__ == "__main__": unittest.main()
