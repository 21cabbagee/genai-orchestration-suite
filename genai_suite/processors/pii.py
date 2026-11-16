"""Rule-based PII redaction run before any text reaches a retriever or LLM."""

import re

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,15}\d\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"(?:\+91[ -]?)?\b[6-9]\d{9}\b|\+\d{1,3}[ -]\d{3}[ -]\d{3}[ -]\d{4}")),
]


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums))
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    found: dict[str, int] = {}
    for label, pattern in PATTERNS:
        def _sub(m: re.Match, label: str = label) -> str:
            if label == "CARD":
                digits = re.sub(r"\D", "", m.group())
                # 12-digit spaced numbers are Aadhaar, not cards; also require a valid Luhn checksum.
                if len(digits) < 13 or not _luhn_ok(digits):
                    return m.group()
            found[label] = found.get(label, 0) + 1
            return f"[{label}]"
        text = pattern.sub(_sub, text)
    return text, found
