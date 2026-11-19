"""Deterministic, evidence-based dataset insights and portable HTML reports."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape

import numpy as np
import pandas as pd


def generate_insights(frame: pd.DataFrame, max_findings: int = 12) -> dict:
    """Summarize data quality and notable patterns without using a language model."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("Upload a dataset with at least one row and one column first.")
    max_findings = max(1, min(int(max_findings), 50))
    rows, column_count = frame.shape
    findings: list[dict[str, str]] = []
    quality: list[dict[str, object]] = []

    duplicate_count = int(frame.duplicated().sum())
    findings.append({
        "category": "Data quality",
        "insight": "Duplicate rows",
        "evidence": f"{duplicate_count:,} duplicate rows out of {rows:,} ({duplicate_count / rows:.1%}).",
    })

    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        distinct = int(series.nunique(dropna=True))
        quality.append({
            "Column": str(column),
            "Type": str(series.dtype),
            "Missing": missing,
            "Missing %": round(missing / rows * 100, 2),
            "Distinct values": distinct,
        })
        if missing:
            findings.append({
                "category": "Data quality",
                "insight": f"Missing values in {column}",
                "evidence": f"{missing:,} of {rows:,} rows ({missing / rows:.1%}) are missing.",
            })
        if distinct == 1 and series.notna().any():
            findings.append({
                "category": "Data quality",
                "insight": f"Constant column: {column}",
                "evidence": "Only one non-missing value appears in this column.",
            })

    numeric = frame.select_dtypes(include="number")
    for column in numeric.columns:
        values = numeric[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        minimum, maximum = float(values.min()), float(values.max())
        q1, q3 = values.quantile([0.25, 0.75]).tolist()
        iqr = q3 - q1
        outlier_count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        findings.append({
            "category": "Numeric summary",
            "insight": f"{column}: center and range",
            "evidence": (
                f"Mean {values.mean():,.4g}; median {values.median():,.4g}; "
                f"range {minimum:,.4g} to {maximum:,.4g} across {len(values):,} non-missing values."
            ),
        })
        if outlier_count:
            findings.append({
                "category": "Numeric pattern",
                "insight": f"Potential outliers in {column}",
                "evidence": f"{outlier_count:,} values fall outside the 1.5×IQR range. Review them; this rule does not prove they are errors.",
            })

    if len(numeric.columns) >= 2:
        corr = numeric.corr(min_periods=3)
        pairs = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value) and abs(float(value)) >= 0.8:
                    pairs.append((abs(float(value)), str(left), str(right), float(value)))
        for _, left, right, value in sorted(pairs, reverse=True)[:5]:
            findings.append({
                "category": "Numeric relationship",
                "insight": f"Strong correlation: {left} and {right}",
                "evidence": f"Pearson correlation is {value:+.2f}. Correlation alone does not establish causation.",
            })

    for column in frame.select_dtypes(exclude="number").columns:
        if not (pd.api.types.is_string_dtype(frame[column]) or pd.api.types.is_object_dtype(frame[column])):
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        top = values.astype(str).value_counts().head(1)
        if not top.empty:
            value, count = str(top.index[0]), int(top.iloc[0])
            if count / len(values) >= 0.5 and values.nunique() > 1:
                findings.append({
                    "category": "Category pattern",
                    "insight": f"Most common value in {column}",
                    "evidence": f"{value!r} appears in {count:,} of {len(values):,} non-missing rows ({count / len(values):.1%}).",
                })

    findings = findings[:max_findings]
    return {
        "rows": int(rows),
        "columns": int(column_count),
        "findings": findings,
        "quality": quality,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_html_report(insights: dict, dataset_name: str = "Uploaded dataset") -> str:
    """Create an offline HTML report with escaped uploaded values and headings."""
    findings = insights.get("findings", [])
    quality = pd.DataFrame(insights.get("quality", []))
    if findings:
        finding_markup = "".join(
            "<li><strong>{}</strong> <span class='tag'>{}</span><p>{}</p></li>".format(
                escape(str(item.get("insight", ""))),
                escape(str(item.get("category", ""))),
                escape(str(item.get("evidence", ""))),
            ) for item in findings
        )
    else:
        finding_markup = "<li>No notable patterns were found by the current checks.</li>"
    table_markup = quality.to_html(index=False, escape=True, border=0, classes="quality") if not quality.empty else "<p>No column details available.</p>"
    title = escape(str(dataset_name))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>InsightForge report — {title}</title><style>
body{{font:16px/1.55 system-ui,Segoe UI,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#18313a;background:#f7faf9}}
h1,h2{{color:#123e48}}.card{{background:white;border:1px solid #d8e5e2;border-radius:12px;padding:20px;margin:16px 0}}
.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.metric{{background:#e9f5f2;padding:12px 18px;border-radius:8px}}
li{{margin:14px 0}}.tag{{font-size:.8em;background:#e7eefb;padding:3px 8px;border-radius:99px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border-bottom:1px solid #d8e5e2;text-align:left;padding:9px}}th{{background:#e9f5f2}}
footer{{color:#52666b;font-size:.9em;margin-top:24px}}
</style></head><body><h1>InsightForge AI — Dataset insights</h1>
<div class="card"><h2>{title}</h2><div class="metrics"><div class="metric"><strong>{int(insights.get('rows', 0)):,}</strong> rows</div>
<div class="metric"><strong>{int(insights.get('columns', 0)):,}</strong> columns</div></div></div>
<section class="card"><h2>Automated findings</h2><p>Descriptive checks computed directly from this dataset; review findings in context.</p><ul>{finding_markup}</ul></section>
<section class="card"><h2>Column quality profile</h2>{table_markup}</section>
<footer>Generated {escape(str(insights.get('generated_at', '')))}. Potential outliers and correlations are signals for review, not proof of an error or causal relationship.</footer>
</body></html>"""
