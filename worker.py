"""Isolated worker for AST-validated, read-only dataframe expressions."""
import json
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_validator import validate_expression


def plain(value):
    if isinstance(value, pd.DataFrame):
        return {"__dataframe__": True, "columns": list(map(str, value.columns)),
                "records": value.astype(object).where(value.notna(), None).to_dict("records")}
    if isinstance(value, pd.Series):
        if len(value) > 1000:
            value = value.head(1000)
        return value.astype(object).where(value.notna(), None).to_dict()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, np.ndarray): return value.tolist()
    return value


def main():
    payload = json.load(sys.stdin)
    frame = pd.DataFrame(payload["records"], columns=payload["columns"])
    for column, dtype in payload.get("dtypes", {}).items():
        if "datetime" in dtype:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    expression = validate_expression(payload["expression"])
    value = eval(compile(expression, "<validated-expression>", "eval"),
                 {"__builtins__": {}, "pd": pd, "np": np, "len": len, "int": int,
                  "float": float, "bool": bool, "abs": abs, "round": round}, {"df": frame})
    print(json.dumps({"ok": True, "value": plain(value)}, default=str, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
