from __future__ import annotations
import math


def validate_result(value, answer_type: str) -> tuple[bool, str]:
    if value is None:
        return False, "The calculation returned no value."
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False, "The calculation returned a non-finite number."
    if answer_type == "boolean" and not isinstance(value, bool):
        return False, f"Expected a Boolean result; received {type(value).__name__}."
    if answer_type == "number" and not isinstance(value, (int, float, dict)):
        return False, f"Expected a number; received {type(value).__name__}."
    if answer_type in {"category", "list[category]", "list[number]"} and not isinstance(value, (str, int, float, list, dict)):
        return False, f"Unexpected result type: {type(value).__name__}."
    return True, "Result type is valid."
