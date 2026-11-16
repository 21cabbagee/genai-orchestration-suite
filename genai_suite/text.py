"""Shared text utilities. Kept deliberately simple so the browser demo can mirror them exactly."""

import re

TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]+\b")  # words, not bare numbers
SENT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "the", "and", "for", "are", "can", "what", "how", "does", "with", "from", "this", "that", "have",
    "has", "was", "were", "will", "into", "your", "you", "our", "who", "when", "need", "needs", "should",
    "must", "may", "per", "any", "all", "not", "but", "its", "their", "them", "they", "his", "her",
    "about", "after", "before", "than", "then", "there", "which", "would", "could", "also", "only",
    "been", "being", "get", "got", "please", "want", "wants", "is", "it", "to", "of", "in", "on", "an",
    "be", "or", "do", "my", "me", "we", "us", "if", "at", "by", "as", "up", "so", "no", "is", "long",
    "much", "many",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stem(word: str) -> str:
    """Tiny suffix stripper (plural, -ed, -ing, trailing e) - enough to match claims/claim, approves/approved."""
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """The analyzer used everywhere: tokenize -> drop stopwords -> stem. Mirrored in docs/app.js."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text.replace("\n", " ")) if s.strip()]
