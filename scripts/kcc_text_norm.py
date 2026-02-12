import re
from typing import Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\u0900-\u097F]")

# Light, language-agnostic normalization for common variants
_VARIANT_RULES = [
    (re.compile(r"\bpm\s*-?\s*kisan\b", re.IGNORECASE), "pmkisan"),
    (re.compile(r"\bpm\s*kisan\s*samman\s*nidhi\b", re.IGNORECASE), "pmkisan"),
    (re.compile(r"\bpradhan\s*mantri\s*kisan\b", re.IGNORECASE), "pmkisan"),
    (re.compile(r"\bpradhan\s*mantri\s*kisan\s*samman\s*nidhi\b", re.IGNORECASE), "pmkisan"),
    (re.compile(r"पीएम\s*किसान"), "pmkisan"),
    (re.compile(r"प्रधान\s*मंत्री\s*किसान"), "pmkisan"),
]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.strip()
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    for pattern, repl in _VARIANT_RULES:
        s = pattern.sub(repl, s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def normalize_many(texts: Iterable[str]) -> list[str]:
    return [normalize_text(t) for t in texts]
