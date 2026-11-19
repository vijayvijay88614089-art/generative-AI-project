from pathlib import Path
import pandas as pd

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "sales_demo.csv"


def demo_frame():
    return pd.read_csv(SAMPLE, parse_dates=["date"])
