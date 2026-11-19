from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

from .code_validator import validate_expression

WORKER = Path(__file__).with_name("worker.py")


def execute_expression(expression: str, frame: pd.DataFrame, timeout: float = 5.0):
    safe_expr = validate_expression(expression)
    payload = {"expression": safe_expr, "columns": list(map(str, frame.columns)),
               "records": frame.astype(object).where(frame.notna(), None).to_dict("records"),
               "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()}}
    try:
        proc = subprocess.run([sys.executable, "-I", str(WORKER)], input=json.dumps(payload, default=str),
                              capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Calculation exceeded the {timeout:g}-second time limit.") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1500:] or "Isolated calculation failed.")
    result = json.loads(proc.stdout)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Calculation failed."))
    value = result.get("value")
    if isinstance(value, dict) and value.get("__dataframe__"):
        return pd.DataFrame(value["records"], columns=value["columns"])
    return value
