from __future__ import annotations
import re

TYPES = ("boolean", "category", "list[category]", "list[number]", "number")


def classify(question: str) -> str:
    q = question.casefold()
    if re.search(r"\b(greater than|less than|above|below|exceed|true or false)\b", q) or re.match(r"^(is|are|does|do|has|have)\b", q.strip()):
        return "boolean"
    if re.search(r"\b(top\s*\d+|first\s*\d+|list|which products|what products|all categories)\b", q):
        return "list[category]"
    if re.search(r"\b(compare|comparison|breakdown)\b", q):
        return "list[number]"
    if re.search(r"\b(highest|lowest|most|least|largest|smallest|best|worst|which region|which product|which category)\b", q):
        return "category"
    if re.search(r"\b(how many|count|total|sum|average|mean|median|minimum|maximum|number of|how much|records|rows|missing)\b", q):
        return "number"
    return "number"
