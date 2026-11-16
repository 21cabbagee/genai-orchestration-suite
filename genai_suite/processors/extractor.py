"""Lightweight structured extraction: amounts, dates, business identifiers and urgency."""

import re

AMOUNT_RE = re.compile(r"(?:INR|Rs\.?|₹|USD|\$)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:rupees|lakh|crore)\b", re.I)
DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b"
    r"|\b(?:today|tomorrow|yesterday|next week|last week)\b", re.I)
ID_RE = re.compile(r"\b(?:INV|PO|ORD|TKT|EMP)-\d{3,}\b", re.I)
DURATION_RE = re.compile(r"\b\d+\s(?:days?|weeks?|months?)\b", re.I)

URGENT = ("urgent", "asap", "immediately", "breach", "chargeback", "lost", "locked", "escalate", "blocked")


def extract_entities(text: str) -> dict[str, list[str]]:
    return {
        "amounts": AMOUNT_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "ids": [i.upper() for i in ID_RE.findall(text)],
        "durations": DURATION_RE.findall(text),
    }


def infer_priority(text: str) -> str:
    lowered = text.lower()
    return "high" if any(word in lowered for word in URGENT) else "normal"
