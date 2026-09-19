"""Cheap local features and source-preserving chunk boundaries."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from functools import lru_cache

# Stop words only affect local retrieval. The semantic judge always receives the original task.
STOP = frozenset(
    """a an the and or of to in on at for from with by as is are was were be been
being it its this that these those i me my we our you your they their he she his her how what
which why when where who do does did can could would should will shall have has had please
find show tell give get all any some about into than then there here explain describe
""".split()
)
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def words(text: str) -> list[str]:
    # Split camelCase as well as snake_case so code and prose can share terms.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


def features(text: str) -> Counter[str]:
    terms = words(text)
    counts = Counter(terms)
    # Bigrams distinguish important local differences such as "not enabled".
    counts.update(a + " " + b for a, b in zip(terms, terms[1:], strict=False))
    return counts


def similarity(a: Counter[str], b: Counter[str], weights: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    intersection = sum(min(v, b.get(k, 0)) * weights.get(k, 1.0) for k, v in a.items())
    total = sum(v * weights.get(k, 1.0) for k, v in a.items())
    total += sum(v * weights.get(k, 1.0) for k, v in b.items())
    return intersection / (total - intersection) if total > intersection else 1.0


@lru_cache(maxsize=65536)
def _feature_bit(term: str) -> int:
    return int.from_bytes(hashlib.blake2s(term.encode(), digest_size=2).digest(), "little") % 8192


def signature(text: str) -> int:
    """Compact lexical sketch used only for fast shortlist diversification."""
    bits = 0
    for term in features(text):
        bits |= 1 << _feature_bit(term)
    return bits


def sketch_similarity(a: int, b: int) -> float:
    union = (a | b).bit_count()
    return (a & b).bit_count() / union if union else 0.0


def chunks(text: str, size: int = 1800, overlap: int = 240):
    """Yield exact (start, end) character ranges; never discard a long record's tail."""
    if size < 8 or not 0 <= overlap < size // 2:
        raise ValueError("chunk size must be >=8 and overlap must be < half the chunk size")
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # Prefer a complete line or sentence near the end of the window.
            floor = start + size * 2 // 3
            boundaries = list(re.finditer(r"\n|(?<=[.!?])\s", text[floor:end]))
            if boundaries:
                end = floor + boundaries[-1].end()
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break
        following = max(start + 1, end - overlap)
        # Start the next window at a word/line boundary when possible.
        boundary = re.search(r"\s", text[following : min(end, following + 80)])
        start = following + boundary.end() if boundary and overlap else following


@lru_cache(maxsize=8)
def encoder(name: str):
    import requests
    import tiktoken

    try:
        try:
            return tiktoken.get_encoding(name)
        except ValueError:
            return tiktoken.encoding_for_model(name)
    except KeyError as e:
        raise ValueError(f"unknown tokenizer {name!r}; use o200k_base, cl100k_base, or bytes") from e
    except requests.RequestException as e:
        raise ValueError(
            "could not download tokenizer data; use --encoding bytes for fully offline use"
        ) from e


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    """Count the exact evidence string. bytes is a conservative, fully offline alternative."""
    if encoding == "bytes":
        return len(text.encode("utf-8"))
    return len(encoder(encoding).encode(text, disallowed_special=()))
