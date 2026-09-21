"""Normalize files and Python iterables into explicit source records."""

from __future__ import annotations

import csv
import fnmatch
import json
import os
import sqlite3
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pathspec import GitIgnoreSpec

from .types import PER, Record

TEXT_FIELDS = ("text", "content", "message", "messages", "conversation", "body", "abstract")
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
SECRET_NAMES = {".env", "credentials.json"}
NON_TEXT = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".zip",
    ".gz",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
}


def field_value(data: Any, name: str) -> Any:
    if isinstance(data, dict) and name in data:
        return data[name]
    value = data
    for part in name.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def per_value(value: Any, where: str) -> str:
    """One representation everywhere: values are compared as text, so 1 and "1" name the same group."""
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"{where}: --per field must be a string or integer")
    return str(value)


def record(
    value: Any,
    *,
    source: str,
    line: int,
    field: str | None = None,
    group_by: str | None = None,
    per: str | None = None,
) -> Record:
    if isinstance(value, Record):
        return value
    if isinstance(value, str):
        if field or group_by or per:
            raise ValueError(f"{source}:{line}: --field, --group-by, and --per require an object")
        return Record(value, id=str(line), source=source, line=line)
    if not isinstance(value, dict):
        raise ValueError(f"{source}:{line}: expected text or an object, got {type(value).__name__}")
    chosen = field or (None if group_by else next((k for k in TEXT_FIELDS if k in value), None))
    try:
        text = as_text(field_value(value, chosen)) if chosen else as_text(value)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError(f"{source}:{line}: missing field {chosen!r}") from e
    metadata = {}
    if group_by:
        try:
            group = field_value(value, group_by)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"{source}:{line}: missing grouping field {group_by!r}") from e
        if not isinstance(group, (str, int)) or isinstance(group, bool):
            raise ValueError(f"{source}:{line}: grouping field must be a string or integer")
        metadata = {"group_key": str(group)}
    if per:
        try:
            part = field_value(value, per)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ValueError(f"{source}:{line}: missing --per field {per!r}") from e
        metadata[PER] = per_value(part, f"{source}:{line}")
    return Record(
        text,
        id=str(value.get("id", value.get("_id", line))),
        source=source,
        line=line,
        field=chosen,
        structured=True,
        metadata=metadata,
    )


def discover(
    paths: Iterable[str | Path], *, glob: str | None = None, exclude: Iterable[str] = ()
) -> Iterator[Path | str]:
    explicit = GitIgnoreSpec.from_lines(exclude)

    def walk(directory: Path, root: Path, rules):
        rules = list(rules)
        for name in (".gitignore", ".ignore"):
            ignore = directory / name
            if ignore.is_file():
                rules.append((directory, GitIgnoreSpec.from_lines(ignore.read_text().splitlines())))
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or path.name in SKIP_DIRS or path.name in SECRET_NAMES:
                continue
            if path.name.startswith(".env.") or path.suffix in {".key", ".jselect"}:
                continue
            is_dir = path.is_dir()
            relative = path.relative_to(root).as_posix()
            if explicit.match_file(relative + ("/" if is_dir else "")):
                continue
            ignored = False
            for base, spec in rules:
                match = spec.check_file(path.relative_to(base).as_posix() + ("/" if is_dir else "")).include
                if match is not None:
                    ignored = match
            if ignored:
                continue
            if is_dir:
                yield from walk(path, root, rules)
            elif not glob or fnmatch.fnmatch(relative, glob) or fnmatch.fnmatch(path.name, glob):
                yield path

    for item in paths:
        if str(item) == "-":
            yield "-"
            continue
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"input does not exist: {item}")
        if path.is_dir():
            rules = []
            parents = []
            if not (path / ".git").exists():
                for parent in path.parents:
                    parents.append(parent)
                    if (parent / ".git").exists():
                        break
                else:
                    parents = []
            for parent in reversed(parents):
                for name in (".gitignore", ".ignore"):
                    p = parent / name
                    if p.is_file():
                        rules.append((parent, GitIgnoreSpec.from_lines(p.read_text().splitlines())))
            yield from walk(path, path, rules)
        else:
            yield path


def _read_paths(
    paths: Iterable[str | Path],
    *,
    field: str | None = None,
    format: str = "auto",
    glob: str | None = None,
    exclude: Iterable[str] = (),
    group_by: str | None = None,
    per: str | None = None,
) -> Iterator[Record]:
    paths = list(paths)
    explicit = {Path(p).expanduser().resolve() for p in paths if str(p) != "-" and Path(p).is_file()}
    for path in discover(paths, glob=glob, exclude=exclude):
        stdin = path == "-"
        source = "(stdin)" if stdin else str(path)
        kind = format
        if kind == "auto":
            suffix = "" if stdin else Path(path).suffix.lower()
            if suffix in NON_TEXT:
                if path in explicit:
                    raise ValueError(f"{source}: unsupported {suffix} input; extract text first")
                continue
            kind = {
                ".jsonl": "jsonl",
                ".ndjson": "jsonl",
                ".json": "json",
                ".csv": "csv",
                ".tsv": "tsv",
                ".log": "lines",
            }.get(suffix, "text")
            if stdin:
                kind = "jsonl" if field else "lines"
        if not stdin:
            with Path(path).open("rb") as probe:
                if b"\x00" in probe.read(8192):
                    if path in explicit:
                        raise ValueError(f"{source}: binary input; extract text first")
                    continue
        stream = sys.stdin if stdin else Path(path).open(encoding="utf-8-sig", newline="")
        try:
            if kind in {"jsonl", "lines"}:
                for line, raw in enumerate(stream, 1):
                    if not raw.strip():
                        continue
                    value = json.loads(raw) if kind == "jsonl" else raw.rstrip("\r\n")
                    yield record(value, source=source, line=line, field=field, group_by=group_by, per=per)
            elif kind == "json":
                data = json.load(stream)
                values = data if isinstance(data, list) else [data]
                for i, value in enumerate(values, 1):
                    rec = record(value, source=source, line=1, field=field, group_by=group_by, per=per)
                    identifier = (
                        str(value.get("id", value.get("_id", i))) if isinstance(value, dict) else str(i)
                    )
                    yield Record(
                        rec.text,
                        identifier,
                        rec.source,
                        rec.line,
                        rec.field,
                        True,
                        {**rec.metadata, **({"json_index": i - 1} if isinstance(data, list) else {})},
                    )
            elif kind in {"csv", "tsv"}:
                reader = csv.reader(stream, delimiter="\t" if kind == "tsv" else ",", strict=True)
                header = next(reader, [])
                if len(header) != len(set(header)):
                    raise ValueError(f"{source}: duplicate column names")
                while True:
                    line = reader.line_num + 1
                    row = next(reader, None)
                    if row is None:
                        break
                    if not row:
                        continue
                    if len(row) != len(header):
                        raise ValueError(f"{source}:{line}: expected {len(header)} columns, got {len(row)}")
                    yield record(
                        dict(zip(header, row, strict=True)),
                        source=source,
                        line=line,
                        field=field,
                        group_by=group_by,
                        per=per,
                    )
            elif kind == "text":
                if field or group_by or per:
                    raise ValueError(f"{source}: --field, --group-by, and --per need JSON/JSONL/CSV input")
                yield Record(stream.read(), id=source, source=source)
            else:
                raise ValueError(f"unknown input format: {kind}")
        except (json.JSONDecodeError, csv.Error, UnicodeError) as e:
            raise ValueError(f"{source}: invalid {kind} input: {e}") from e
        finally:
            if not stdin:
                stream.close()


def group_records(data: Iterable[Record], group_by: str) -> Iterator[Record]:
    """Group nonadjacent rows using a disk-backed spool, preserving input order."""
    db = sqlite3.connect("")
    try:
        db.execute("CREATE TABLE rows (ordinal INTEGER PRIMARY KEY, group_key TEXT, payload TEXT)")
        for i, rec in enumerate(data):
            if "group_key" not in rec.metadata:
                raise ValueError(f"{rec.source}:{rec.line}: Record metadata needs group_key for grouping")
            db.execute(
                "INSERT INTO rows VALUES(?,?,?)",
                (i, rec.metadata["group_key"], json.dumps(asdict(rec), ensure_ascii=False)),
            )
        db.execute("CREATE INDEX groups ON rows(group_key, ordinal)")
        for (key,) in db.execute("SELECT group_key FROM rows GROUP BY group_key ORDER BY MIN(ordinal)"):
            texts, members, position, parts = [], [], 0, set()
            for (payload,) in db.execute(
                "SELECT payload FROM rows WHERE group_key=? ORDER BY ordinal", (key,)
            ):
                row = json.loads(payload)
                text = row["text"]
                parts.add(row["metadata"].get(PER))
                members.append(
                    {
                        "source": row["source"],
                        "record_id": row["id"],
                        "line": row["line"],
                        "field": row["field"],
                        "start": position,
                        "end": position + len(text),
                        **{k: v for k, v in row["metadata"].items() if k not in {"group_key", PER}},
                    }
                )
                texts.append(text)
                position += len(text) + 2
            if len(parts) > 1:
                raise ValueError(f"group {key!r} spans more than one --per value; its rows must agree")
            metadata = {"group_by": group_by, "group_id": key, "members": members}
            if parts != {None}:
                metadata[PER] = parts.pop()
            yield Record(
                "\n\n".join(texts), id=key, source="(grouped records)", structured=True, metadata=metadata
            )
    finally:
        db.close()


def read_paths(paths, *, group_by: str | None = None, **options) -> Iterator[Record]:
    raw = _read_paths(paths, group_by=group_by, **options)
    yield from group_records(raw, group_by) if group_by else raw


def records(
    data, *, field: str | None = None, group_by: str | None = None, per: str | None = None, **options
) -> Iterator[Record]:
    if isinstance(data, (str, os.PathLike)):
        yield from read_paths([data], field=field, group_by=group_by, per=per, **options)
    elif isinstance(data, Record):
        yield from group_records([data], group_by) if group_by else [data]
    else:
        raw = (
            record(value, source="(records)", line=i, field=field, group_by=group_by, per=per)
            for i, value in enumerate([data] if isinstance(data, dict) else data, 1)
        )
        yield from group_records(raw, group_by) if group_by else raw
