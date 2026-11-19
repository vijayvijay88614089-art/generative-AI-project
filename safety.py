from __future__ import annotations
import re

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions", r"reveal (the )?(system|hidden) prompt",
    r"show me your chain.of.thought", r"override (your )?(rules|instructions)",
    r"developer message", r"system prompt", r"exfiltrate", r"send .* (api key|password|secret)",
]


def detect_prompt_injection(text: str) -> list[str]:
    found = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            found.append(pattern)
    return found
