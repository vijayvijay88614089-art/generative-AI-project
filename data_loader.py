from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import pandas as pd

MAX_ROWS = 100_000
MAX_COLS = 100
MAX_CELLS = 2_000_000
MAX_FILE_BYTES = 50 * 1024 * 1024


def load_table(upload) -> pd.DataFrame:
    """Read an uploaded CSV/XLSX or path, enforce predictable resource bounds."""
    name = getattr(upload, "name", str(upload))
    suffix = Path(name).suffix.lower()
    if hasattr(upload, "getvalue"):
        payload = upload.getvalue()
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError(f"This demo accepts files up to {MAX_FILE_BYTES // (1024 * 1024)} MB.")
        source = BytesIO(payload)
    else:
        source = name
        path = Path(name)
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"This demo accepts files up to {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    if suffix == ".csv":
        frame = pd.read_csv(source, nrows=MAX_ROWS + 1)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(source, nrows=MAX_ROWS + 1)
    else:
        raise ValueError("Supported dataset formats are CSV and Excel (.xlsx/.xls).")
    if frame.empty and len(frame.columns) == 0:
        raise ValueError("The file is empty or has no readable header row.")
    if frame.empty:
        raise ValueError("The dataset has column headers but no data rows.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"This demo is limited to {MAX_ROWS:,} rows per upload.")
    if len(frame.columns) > MAX_COLS:
        raise ValueError(f"This demo is limited to {MAX_COLS} columns per upload.")
    if frame.size > MAX_CELLS:
        raise ValueError(f"This demo is limited to {MAX_CELLS:,} cells per upload for responsive analysis.")
    names = [str(c).strip() for c in frame.columns]
    for index, name in enumerate(names):
        # Pandas represents blank CSV header cells as "Unnamed: n".
        if not name or name.casefold().startswith("unnamed:"):
            names[index] = f"column_{index + 1}"
    frame.columns = names
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique after trimming whitespace.")
    for col in frame.columns:
        dtype_name = str(frame[col].dtype).casefold()
        if dtype_name not in {"object", "str", "string"}:
            continue
        sample = frame[col].dropna().astype(str).head(10)
        looks_like_date = "date" in col.casefold() or "time" in col.casefold() or (
            len(sample) > 0 and sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > .8
        )
        if not looks_like_date:
            continue
        parsed = pd.to_datetime(frame[col], errors="coerce")
        if len(frame) and parsed.notna().mean() > .85:
            frame[col] = parsed
    return frame


def profile(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "column_names": frame.columns.astype(str).tolist(),
        "types": {str(k): str(v) for k, v in frame.dtypes.items()},
        "missing": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "duplicates": int(frame.duplicated().sum()),
        "numeric": frame.select_dtypes(include="number").columns.tolist(),
        "categorical": frame.select_dtypes(exclude="number").columns.tolist(),
        "sample": frame.head(5).astype(object).where(frame.head(5).notna(), None).to_dict("records"),
    }
