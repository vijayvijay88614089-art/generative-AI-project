from __future__ import annotations
from math import isclose
import pandas as pd
from .analyst import analyze
from .database import initialize, record_evaluation


DEMO_CASES = [
    {"question":"How many rows are in this dataset?", "expected_type":"number", "expected":24},
    {"question":"What is the average unit_price?", "expected_type":"number", "expected":19.125},
    {"question":"Which region has the highest revenue?", "expected_type":"category", "expected":"East"},
    {"question":"Show top 3 products by revenue", "expected_type":"list[category]", "expected":["Headphones","Desk Lamp","Notebook"]},
    {"question":"How many missing values are in revenue?", "expected_type":"number", "expected":0},
    {"question":"Is total revenue greater than 4000?", "expected_type":"boolean", "expected":True},
]


def equal(actual, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    return actual == expected


def run_evaluation(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    initialize()
    rows = []
    for case in DEMO_CASES:
        result = analyze(case["question"], frame)
        rows.append({"Question":case["question"], "Expected type":case["expected_type"],
                     "Predicted type":result.get("answer_type"), "Expected":str(case["expected"]),
                     "Actual":str(result.get("value", result.get("error"))), "Exact match":bool(result.get("ok") and equal(result.get("value"),case["expected"])),
                     "Execution success":bool(result.get("ok")), "Type valid":result.get("answer_type")==case["expected_type"],
                     "Seconds":result.get("elapsed_seconds",0), "Corrections":result.get("attempts",0)})
    table = pd.DataFrame(rows)
    count = len(table)
    metrics = {"cases":count, "exact_match_accuracy":float(table["Exact match"].mean()),
               "execution_success_rate":float(table["Execution success"].mean()),
               "answer_type_validation_accuracy":float(table["Type valid"].mean()),
               "mean_response_seconds":float(table["Seconds"].mean()),
               "error_correction_success_rate":None,
               "error_correction_note":"No correction attempts occurred in this deterministic demo set." if not table["Corrections"].sum() else "Measured among attempted corrections."}
    if table["Corrections"].sum():
        metrics["error_correction_success_rate"] = float(table.loc[table.Corrections > 0, "Execution success"].mean())
    record_evaluation(metrics)
    return table, metrics
