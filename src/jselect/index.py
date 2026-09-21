"""A disk-backed, reusable candidate index. No model is called during indexing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path

from .inputs import records
from .text import chunks, signature, sketch_similarity, words
from .types import Passage, merge_passages

SCHEMA = 1


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Index:
    """Use `Index.build(records)` for a temporary index, or supply path= for a saved one."""

    def __init__(self, path: str | Path, *, temporary: bool = False):
        self.path = Path(path)
        self.temporary = temporary
        self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            self.stats = json.loads(self.db.execute("SELECT value FROM meta WHERE key='stats'").fetchone()[0])
            if self.stats["schema_version"] != SCHEMA:
                raise ValueError("unsupported jselect index version; rebuild the index")
        except (sqlite3.DatabaseError, TypeError, KeyError, ValueError) as e:
            self.db.close()
            raise ValueError(f"not a jselect index: {self.path}") from e

    @classmethod
    def build(
        cls,
        data,
        *,
        path: str | Path | None = None,
        field: str | None = None,
        group_by: str | None = None,
        chunk_size: int = 1800,
        overlap: int = 240,
        force: bool = False,
    ) -> Index:
        # Validate even when input is empty.
        if chunk_size < 64:
            raise ValueError("index chunk size must be >=64")
        list(chunks("", chunk_size, overlap))
        target = Path(path).expanduser().resolve() if path else None
        if target and target.exists() and not force:
            raise ValueError(f"index already exists: {target}; use --force to replace it")
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=".jselect-", suffix=".sqlite", dir=target.parent if target else None
        )
        os.close(fd)
        db = sqlite3.connect(name)
        counts = {
            "schema_version": SCHEMA,
            "records": 0,
            "passages": 0,
            "unique_passages": 0,
            "empty_records": 0,
            "characters": 0,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        try:
            db.executescript("""
                CREATE TABLE passages (id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL, sources TEXT NOT NULL, occurrences INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA cache_size=-16000;
            """)
            db.execute("BEGIN")
            for rec in records(data, field=field, group_by=group_by):
                counts["records"] += 1
                counts["characters"] += len(rec.text)
                if not rec.text.strip():
                    counts["empty_records"] += 1
                record_digest = _digest(rec.text)
                preceding_end, preceding_line = 0, rec.line
                for start, end in chunks(rec.text, chunk_size, overlap):
                    text = rec.text[start:end]
                    key = _digest(text)
                    # Count incrementally; overlapping passages can start before the prior end.
                    line = (
                        rec.line
                        if rec.structured
                        else preceding_line + rec.text[preceding_end:start].count("\n")
                    )
                    if start < preceding_end and not rec.structured:
                        line = rec.line + rec.text[:start].count("\n")
                    preceding_end, preceding_line = start, line
                    ref = {
                        **{k: v for k, v in rec.metadata.items() if k != "members"},
                        "source": rec.source,
                        "record_id": rec.id,
                        "line": line,
                        "start": start,
                        "end": end,
                        "field": rec.field,
                        "record_sha256": record_digest,
                        "passage_id": key,
                        "structured": rec.structured,
                    }
                    if "members" in rec.metadata:
                        ref["members"] = [
                            m for m in rec.metadata["members"] if m["end"] > start and m["start"] < end
                        ]
                    ref["end_line"] = line if rec.structured else line + text.rstrip("\n").count("\n")
                    old = db.execute("SELECT id, sources FROM passages WHERE key=?", (key,)).fetchone()
                    if old:
                        sources = json.loads(old[1])
                        if len(sources) < 5 and ref not in sources:
                            sources.append(ref)
                        db.execute(
                            "UPDATE passages SET occurrences=occurrences+1, sources=? WHERE id=?",
                            (json.dumps(sources, ensure_ascii=False), old[0]),
                        )
                    else:
                        db.execute(
                            "INSERT INTO passages(key,text,sources) VALUES(?,?,?)",
                            (key, text, json.dumps([ref], ensure_ascii=False)),
                        )
                        counts["unique_passages"] += 1
                    counts["passages"] += 1
            db.execute(
                "CREATE VIRTUAL TABLE search USING fts5(text, content=passages, content_rowid=id, "
                "tokenize='porter unicode61')"
            )
            db.execute("INSERT INTO search(search) VALUES('rebuild')")
            db.execute("INSERT INTO meta VALUES ('stats', ?)", (json.dumps(counts),))
            db.commit()
            db.close()
            if target:
                os.replace(name, target)
            return cls(target or name, temporary=target is None)
        except BaseException:
            db.close()
            Path(name).unlink(missing_ok=True)
            raise

    def close(self):
        self.db.close()
        if self.temporary:
            self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _passage(row, score: float = 0.0) -> Passage:
        sources = json.loads(row["sources"])
        for ref in sources:
            ref.setdefault("passage_id", row["key"])
        return Passage(row["key"], row["text"], sources, row["occurrences"], score)

    def get(self, id: str) -> Passage:
        row = self.db.execute("SELECT * FROM passages WHERE key=?", (id,)).fetchone()
        if row is None:
            raise ValueError(
                f"no indexed passage with id {id!r}; for fitted excerpts, use sources[].passage_id"
            )
        return self._passage(row)

    def all(self):
        for row in self.db.execute("SELECT * FROM passages ORDER BY id"):
            yield self._passage(row)

    @staticmethod
    def _expression(task: str) -> str:
        terms = list(dict.fromkeys(words(task)))[:64]
        if not terms:
            terms = [w for w in task.split() if w.strip()][:64]
        return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def matches(self, task: str):
        """Every passage sharing a task term, best first, with no shortlist: the local sampling population."""
        expression = self._expression(task)
        best = None
        for row in (
            self.db.execute(
                "SELECT p.*, bm25(search) AS rank FROM search JOIN passages p ON p.id=search.rowid "
                "WHERE search MATCH ? ORDER BY rank, p.id",
                (expression,),
            )
            if expression
            else []
        ):
            best = -row["rank"] if best is None else best
            yield self._passage(row, max(0.0, -row["rank"] / best))

    def search(self, task: str, limit: int = 256, *, exclude: set[str] | None = None) -> list[Passage]:
        """Plain BM25 results, for callers that want retrieval without diversification or model calls."""
        if limit <= 0:
            raise ValueError("candidate limit must be positive")
        exclude = exclude or set()
        expression = self._expression(task)
        rows = (
            self.db.execute(
                "SELECT p.*, bm25(search) AS rank FROM search JOIN passages p ON p.id=search.rowid "
                "WHERE search MATCH ? ORDER BY rank, p.id LIMIT ?",
                (expression, limit + len(exclude)),
            ).fetchall()
            if expression
            else []
        )
        rows = [row for row in rows if row["key"] not in exclude][:limit]
        best = -rows[0]["rank"] if rows else 1.0
        return [self._passage(row, max(0.0, -row["rank"] / best)) for row in rows]

    def candidates(
        self, task: str, limit: int = 256, *, explore: float = 0.2, exclude: set[str] | None = None
    ) -> list[Passage]:
        if not 0 <= explore <= 1:
            raise ValueError("explore must be between 0 and 1")
        exclude = exclude or set()
        lexical = self.search(task, limit * 4, exclude=exclude)
        # Use a modest diversity penalty before semantic scoring so common near-duplicates
        # are less likely to consume the entire shortlist.
        reserve = math.ceil(limit * explore)
        selected = self._spread(lexical, max(0, limit - reserve))
        seen = {p.id for p in selected} | exclude
        # Content hashes provide a deterministic, query-seeded sample without sorting
        # the whole collection by RANDOM() on every query.
        pivot = _digest(task)
        for operator, bound in ((">=", pivot), ("<", pivot)):
            for row in self.db.execute(
                f"SELECT * FROM passages WHERE key {operator} ? ORDER BY key LIMIT ?",
                (bound, limit + len(exclude)),
            ):
                if len(selected) >= limit:
                    break
                if row["key"] not in seen:
                    passage = next((p for p in lexical if p.id == row["key"]), None)
                    selected.append(passage or self._passage(row))
                    seen.add(row["key"])
        return selected

    @staticmethod
    def _spread(passages: list[Passage], limit: int) -> list[Passage]:
        passages = merge_passages(passages)
        if len(passages) <= limit:
            return passages
        # Linear candidate-by-selected work; all expensive operations are bounded by shortlist size.
        vectors = [signature(p.text) for p in passages]
        redundant = [0.0] * len(passages)
        remaining = set(range(len(passages)))
        chosen = []
        while remaining and len(chosen) < limit:
            best = max(remaining, key=lambda i: (passages[i].retrieval_score * (1 - 0.6 * redundant[i]), -i))
            chosen.append(passages[best])
            remaining.remove(best)
            for i in remaining:
                redundant[i] = max(redundant[i], sketch_similarity(vectors[i], vectors[best]))
        return chosen

    def select(self, *, task: str, tokens: int = 8000, **options):
        from .select import select

        return select(self, task=task, tokens=tokens, **options)
