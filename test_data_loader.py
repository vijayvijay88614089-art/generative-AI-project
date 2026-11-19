import unittest
from io import BytesIO
import pandas as pd
from modules.data_loader import load_table
from tests.helpers import SAMPLE


class UploadedFixture:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


class DataLoaderTests(unittest.TestCase):
    def test_loads_demo_csv_and_detects_date(self):
        frame = load_table(str(SAMPLE))
        self.assertEqual(frame.shape, (24, 8))
        self.assertTrue(str(frame["date"].dtype).startswith("datetime64["))

    def test_loads_browser_uploaded_csv_and_excel(self):
        source = pd.DataFrame({"region": ["East", "West"], "revenue": [12, 7]})
        csv = source.to_csv(index=False).encode("utf-8")
        xlsx_buffer = BytesIO()
        source.to_excel(xlsx_buffer, index=False)
        for upload in (
            UploadedFixture("data.csv", csv),
            UploadedFixture("data.xlsx", xlsx_buffer.getvalue()),
        ):
            with self.subTest(filename=upload.name):
                frame = load_table(upload)
                self.assertEqual(frame.shape, (2, 2))
                self.assertEqual(frame["revenue"].tolist(), [12, 7])

    def test_rejects_header_only_and_names_blank_columns(self):
        with self.assertRaisesRegex(ValueError, "no data rows"):
            load_table(UploadedFixture("empty.csv", b"region,revenue\n"))
        frame = load_table(UploadedFixture("blank_header.csv", b",revenue\nEast,12\n"))
        self.assertEqual(frame.columns.tolist(), ["column_1", "revenue"])


if __name__ == "__main__": unittest.main()
