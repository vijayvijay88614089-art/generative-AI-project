import unittest
from modules.code_validator import validate_expression


class CodeValidatorTests(unittest.TestCase):
    def test_accepts_safe_dataframe_expression(self):
        expression = "df.groupby('region')['revenue'].sum().idxmax()"
        self.assertEqual(validate_expression(expression), expression)

    def test_rejects_import_and_private_attributes(self):
        for expression in ["__import__('os').system('whoami')", "df.__class__", "open('secret.txt')"]:
            with self.subTest(expression=expression), self.assertRaises(ValueError): validate_expression(expression)

    def test_rejects_statements(self):
        with self.assertRaises(ValueError): validate_expression("x = 1")


if __name__ == "__main__": unittest.main()
