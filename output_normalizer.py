from __future__ import annotations
import math
import numpy as np
import pandas as pd


def normalize(value):
    """Convert pandas/numpy scalars and containers into stable JSON-friendly values."""
    if value is pd.NA or value is pd.NaT: return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, pd.Series): return {str(k): normalize(v) for k,v in value.head(1000).items()}
    if isinstance(value, pd.DataFrame): return value.head(1000).to_dict("records")
    if isinstance(value, dict): return {str(k): normalize(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [normalize(v) for v in value]
    return value
