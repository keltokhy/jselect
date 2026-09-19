"""Public records and results. Excerpts always retain their original text."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from typing import Any


@dataclass(frozen=True)
class Record:
    text: str
    id: str = ""
    source: str = "(records)"
    line: int = 1
    field: str | None = None
    # In a structured field, embedded newlines do not change the file's row location.
    structured: bool = False
    metadata: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    sources: list[dict[str, Any]]
    occurrences: int = 1
    retrieval_score: float = 0.0


def merge_passages(passages: list[Passage]) -> list[Passage]:
    """Coalesce identical fitted text while retaining known provenance from each parent."""
    merged: dict[str, Passage] = {}
    for p in passages:
        if p.text not in merged:
            merged[p.text] = p
            continue
        old = merged[p.text]
        sources = list(old.sources)
        for ref in p.sources:
            if len(sources) < 5 and ref not in sources:
                sources.append(ref)
        merged[p.text] = Passage(
            old.id,
            old.text,
            sources,
            old.occurrences + p.occurrences,
            max(old.retrieval_score, p.retrieval_score),
        )
    return list(merged.values())


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    sources: list[dict[str, Any]]
    occurrences: int
    relevance: float
    novelty: float
    reason: str


@dataclass
class Selection:
    task: str
    context: str
    items: list[Evidence]
    tokens: int
    token_budget: int
    encoding: str
    stats: dict[str, Any]
    warnings: list[str] = dc_field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
