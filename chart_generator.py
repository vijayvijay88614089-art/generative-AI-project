from __future__ import annotations
import pandas as pd
import plotly.express as px


def build_chart(frame: pd.DataFrame, question: str, result):
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    q = question.casefold()
    if any(term in q for term in ("correlation", "correlated", "relationship between")) and len(numeric) >= 2:
        return px.imshow(frame[numeric].corr(numeric_only=True), text_auto=".2f", aspect="auto", title="Numeric correlations")
    measure = next((c for c in numeric if str(c).casefold() in q), numeric[0] if numeric else None)
    group = next((c for c in categorical if str(c).casefold() in q), None)
    if group and measure and any(term in q for term in ("compare", "by ", "per ", "top", "highest", "lowest", "most", "least", "revenue", "sales", "total", "sum", "distribution", "breakdown")):
        data = frame.groupby(group, dropna=False)[measure].sum().reset_index().sort_values(measure, ascending=False).head(20)
        return px.bar(data, x=group, y=measure, title=f"{measure} by {group}")
    dates = [c for c in frame.columns if "date" in str(c).casefold() or "time" in str(c).casefold()]
    if dates and measure and any(x in q for x in ("trend", "month", "over time")):
        series = frame.assign(_period=pd.to_datetime(frame[dates[0]], errors="coerce").dt.to_period("M").astype(str))
        data = series.groupby("_period")[measure].sum().reset_index()
        return px.line(data, x="_period", y=measure, markers=True, title=f"Monthly {measure} trend")
    return None
