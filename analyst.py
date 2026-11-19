from __future__ import annotations
import time

from .code_generator import generate_expression
from .code_executor import execute_expression
from .error_corrector import correct_expression
from .explanation import explain
from .metadata import build_metadata
from .result_validator import validate_result
from .code_validator import validate_expression
from .output_normalizer import normalize


def analyze(question: str, frame, max_retries: int = 3, rag_context: str = "",
            local_model_id: str | None = None) -> dict:
    start = time.perf_counter()
    meta = build_metadata(frame)
    try:
        expr, answer_type, examples, source = generate_expression(
            question, frame, meta, rag_context, local_model_id=local_model_id
        )
    except Exception as exc:
        return {"ok": False,
                "error": f"The selected language model could not generate a result: {type(exc).__name__}: {exc}",
                "answer_type": "unknown", "expression": "", "attempts": 0,
                "examples": [], "source": "Model unavailable",
                "elapsed_seconds": time.perf_counter() - start, "metadata": meta}
    attempts = 0
    error = None
    if not expr:
        if source == "Clarification required":
            numeric_columns = ", ".join(map(str, meta.get("numeric", [])))
            return {"ok": False,
                    "error": f"Please specify which numeric column to use ({numeric_columns}) and ask again.",
                    "answer_type": answer_type, "expression": "", "attempts": 0,
                    "examples": examples, "source": source,
                    "elapsed_seconds": time.perf_counter() - start, "metadata": meta}
        return {"ok": False,
                "error": "This question is outside the local planner. Try a row count, missing-value count, total, average, or comparison by a category; enable Qwen or configure Groq for broader language understanding.",
                "answer_type": answer_type, "expression": "", "attempts": 0,
                "examples": examples, "source": source,
                "elapsed_seconds": time.perf_counter() - start, "metadata": meta}
    while True:
        try:
            validate_expression(expr)
            value = normalize(execute_expression(expr, frame))
            valid, message = validate_result(value, answer_type)
            if not valid:
                raise ValueError(message)
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempts >= max_retries or not __import__("os").getenv("GROQ_API_KEY"):
                return {"ok": False, "error": error, "answer_type": answer_type, "expression": expr,
                        "attempts": attempts, "examples": examples, "source": source,
                        "elapsed_seconds": time.perf_counter() - start, "metadata": meta}
            attempts += 1
            try:
                expr = correct_expression(question, expr, error, list(frame.columns))
                source = "Groq multi-turn correction"
            except Exception as correction_error:
                error = f"{error}; correction unavailable: {correction_error}"
                return {"ok": False, "error": error, "answer_type": answer_type, "expression": expr,
                        "attempts": attempts, "examples": examples, "source": source,
                        "elapsed_seconds": time.perf_counter() - start, "metadata": meta}
    explanation, explanation_source = explain(question, value, expr, rag_context)
    return {"ok": True, "value": value, "answer_type": answer_type, "expression": expr,
            "attempts": attempts, "examples": examples, "source": source,
            "explanation": explanation, "explanation_source": explanation_source,
            "elapsed_seconds": time.perf_counter() - start, "metadata": meta}
