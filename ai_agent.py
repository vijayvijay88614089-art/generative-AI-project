from __future__ import annotations
from .analyst import analyze
from .chart_generator import build_chart
from .data_loader import profile
from .rag_pipeline import retrieve

TOOLS = {
    "dataset summary": "Read-only row/column/missing-value summary",
    "statistical analysis": "Read-only summary and aggregation",
    "grouping and aggregation": "Allowlisted pandas groupby aggregation",
    "chart generation": "Plot-only chart from the uploaded table",
    "document retrieval": "Read-only retrieval from uploaded PDF/TXT chunks",
}


def plan(question: str) -> list[dict]:
    q = question.casefold()
    steps = []
    if any(x in q for x in ("document", "paper", "according to", "pdf", "file")):
        steps.append({"tool": "document retrieval", "reason": "Question asks about reference material."})
    if any(x in q for x in ("plot", "chart", "graph", "trend", "visualize", "visualise")):
        steps.append({"tool": "chart generation", "reason": "Question requests a visual result."})
    if any(x in q for x in ("highest", "lowest", "top", "by region", "by product", "compare")):
        steps.append({"tool": "grouping and aggregation", "reason": "Question compares grouped values."})
    elif any(x in q for x in ("average", "mean", "sum", "total", "how many", "missing")):
        steps.append({"tool": "statistical analysis", "reason": "Question requests a numeric or quality calculation."})
    if any(x in q for x in ("dataset summary", "data profile", "schema", "columns", "missing values")) and not any(s["tool"] in {"statistical analysis", "grouping and aggregation"} for s in steps):
        steps.append({"tool": "dataset summary", "reason": "Question asks to inspect the uploaded table."})
    if not steps:
        steps.append({"tool": "dataset summary", "reason": "Fallback safe inspection."})
    return steps


def run(question: str, frame, rag_chunks: list | None = None):
    """Execute the visible read-only plan and report each tool's actual outcome."""
    plan_steps = plan(question)
    result = None
    chart = None
    evidence = []
    actions = []
    for step in plan_steps:
        tool = step["tool"]
        if tool in {"statistical analysis", "grouping and aggregation"}:
            if result is None:
                result = analyze(question, frame)
            actions.append({"tool": tool, "status": "completed" if result["ok"] else "failed",
                            "details": "Validated local calculation" if result["ok"] else result["error"]})
        elif tool == "dataset summary":
            summary = profile(frame)
            actions.append({"tool": tool, "status": "completed",
                            "details": f"{summary['rows']} rows, {summary['columns']} columns, {summary['duplicates']} duplicate rows"})
        elif tool == "chart generation":
            chart = build_chart(frame, question, result.get("value") if result and result.get("ok") else None)
            actions.append({"tool": tool, "status": "completed" if chart is not None else "unavailable",
                            "details": "Chart created from the active dataset" if chart is not None else "No suitable chart could be inferred from this question"})
        elif tool == "document retrieval":
            evidence = retrieve(rag_chunks or [], question)
            actions.append({"tool": tool, "status": "completed" if evidence else "unavailable",
                            "details": f"Retrieved {len(evidence)} relevant passages" if evidence else "No indexed document passages are available or relevant"})
    return {"plan": plan_steps, "actions": actions, "result": result, "chart": chart, "evidence": evidence}
