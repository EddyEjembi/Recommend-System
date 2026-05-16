"""Text cleaning helpers for ingested review content."""

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Return a normalised, single-spaced version of `text`."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_pii(text: str) -> str:
    """Remove obvious PII patterns from review text.

    Currently strips email addresses and 10-digit phone numbers; extend as
    additional patterns are discovered in the dataset.
    """
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[phone]", text)
    return text
