import re
from collections import Counter

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "he", "her", "his", "i", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "will", "with",
    "you", "your", "they", "them", "we", "our", "about", "into", "than", "then"
}


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation while keeping words readable."""
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text into normalized alpha tokens, filtering stopwords."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens = normalized.split()
    return [token for token in tokens if token and token not in STOPWORDS and token.isalpha()]


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Return the most common meaningful words as a simple keyword list."""
    tokens = tokenize(text)
    if not tokens:
        return []
    counts = Counter(tokens)
    return [term for term, _ in counts.most_common(top_n)]
