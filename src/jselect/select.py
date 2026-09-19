"""Task-aware evidence selection with explicit retrieval and output budgets."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from collections import Counter
from dataclasses import replace
from itertools import islice
from pathlib import Path

from .index import Index, _digest
from .inputs import records
from .judge import JevScorer, SemanticError, resolve_backend, validate_score
from .text import chunks, count_tokens, features, similarity, words
from .types import Evidence, Passage, Selection, merge_passages


def render_item(item: Evidence | Passage, number: int) -> str:
    ref = item.sources[0]
    location = {
        "source": ref["source"],
        "line": ref["line"],
        "id": ref["record_id"],
        "chars": [ref["start"], ref["end"]],
    }
    if ref.get("field"):
        location["field"] = ref["field"]
    if "json_index" in ref:
        location["json_index"] = ref["json_index"]
    if "group_id" in ref:
        location["group_by"] = ref["group_by"]
        location["group_id"] = ref["group_id"]
        location["rows"] = [{"source": m["source"], "line": m["line"]} for m in ref.get("members", [])]
    return f"[{number}] {json.dumps(location, ensure_ascii=False, separators=(',', ':'))}\n{item.text}"


def render(items) -> str:
    return "\n\n".join(render_item(item, i + 1) for i, item in enumerate(items))


def previous_texts(against) -> list[str]:
    if against is None:
        return []
    if isinstance(against, Selection):
        return [item.text for item in against.items]
    if isinstance(against, (str, Path)):
        path = Path(against)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "items" in data and "schema_version" in data:
                return [item["text"] for item in data["items"]]
        return [rec.text for rec in records(path)]
    if isinstance(against, dict) and "items" in against and "schema_version" in against:
        return [item["text"] for item in against["items"]]
    return [rec.text for rec in records(against)]


def fit_passages(
    passages: list[Passage], *, tokens: int, encoding: str, task: str, local: bool = False
) -> list[Passage]:
    """Rechunk oversized candidates before judging, so scores describe exactly the returned evidence."""
    result = []
    query = set(words(task))
    for p in passages:
        if count_tokens(render_item(p, 1), encoding) <= tokens:
            result.append(p)
            continue
        overhead = count_tokens(render_item(Passage(p.id, "", p.sources), 1), encoding)
        allowance = tokens - overhead - 8
        if allowance < 16:
            continue
        body_tokens = count_tokens(p.text, encoding)
        size = max(8, min(len(p.text) - 1, int(len(p.text) * allowance / max(body_tokens, 1) * 0.75)))
        for start, end in chunks(p.text, size, min(size // 5, 160)):
            text = p.text[start:end]
            refs = []
            for old in p.sources:
                ref = dict(old)
                ref.update(start=old["start"] + start, end=old["start"] + end)
                if "members" in old:
                    ref["members"] = [
                        m for m in old["members"] if m["end"] > ref["start"] and m["start"] < ref["end"]
                    ]
                if not old.get("structured"):
                    ref["line"] += p.text[:start].count("\n")
                    ref["end_line"] = ref["line"] + text.rstrip("\n").count("\n")
                refs.append(ref)
            score = p.retrieval_score
            if local:
                score *= len(query & set(words(text))) / max(len(query), 1)
            part = Passage(_digest(text), text, refs, p.occurrences, score)
            if count_tokens(render_item(part, 1), encoding) <= tokens:
                result.append(part)
            elif len(part.text) < len(p.text) and len(part.text) > 8:
                result.extend(fit_passages([part], tokens=tokens, encoding=encoding, task=task, local=local))
    return merge_passages(result)


def pack(
    passages: list[Passage],
    scores: list[float],
    *,
    tokens: int,
    encoding: str,
    diversity: float,
    threshold: float,
    previous: list[str],
    max_items: int | None = None,
    mode: str = "semantic",
) -> list[Evidence]:
    """Greedy relevance/novelty per token. Arithmetic and provenance stay in code."""
    if len(scores) != len(passages):
        raise ValueError("scorer must return exactly one score per passage")
    scores = [validate_score(score) for score in scores]
    already = set(previous)
    eligible = [
        i
        for i, p in enumerate(passages)
        if scores[i] >= threshold and scores[i] > 0 and p.text not in already
    ]
    vectors = {i: features(passages[i].text) for i in eligible}
    prior_vectors = [features(text) for text in previous]
    df = Counter(k for vector in [*vectors.values(), *prior_vectors] for k in vector)
    weights = {k: 1 + math.log((len(vectors) + len(prior_vectors) + 1) / (n + 1)) for k, n in df.items()}
    redundancy = {
        i: max((similarity(vectors[i], v, weights) for v in prior_vectors), default=0.0) for i in eligible
    }
    costs = {i: count_tokens(render_item(passages[i], 1), encoding) for i in eligible}
    eligible = {i for i in eligible if costs[i] <= tokens}
    selected: list[Evidence] = []
    remaining = tokens
    while eligible and (max_items is None or len(selected) < max_items):
        best = max(
            eligible,
            key=lambda i: (
                scores[i] * (1 - diversity * redundancy[i]) / max(costs[i], 1) ** 0.25,
                scores[i],
                -i,
            ),
        )
        eligible.remove(best)
        p = passages[best]
        item = Evidence(
            p.id,
            p.text,
            p.sources,
            p.occurrences,
            scores[best],
            1 - redundancy[best],
            f"{mode} relevance; greedy selection balances relevance, text novelty, and token cost",
        )
        proposed = [*selected, item]
        # Count the final serialized context, including citations and separators.
        actual = count_tokens(render(proposed), encoding)
        if actual > tokens:
            continue
        selected = proposed
        eligible = {i for i in eligible if passages[i].text != p.text}
        remaining = tokens - actual
        if remaining <= 0:
            break
        for i in eligible:
            redundancy[i] = max(redundancy[i], similarity(vectors[i], vectors[best], weights))
    return selected


async def aselect(
    data,
    *,
    task: str,
    tokens: int = 8000,
    mode: str = "auto",
    candidates: int = 256,
    encoding: str = "o200k_base",
    diversity: float = 0.7,
    threshold: float | None = None,
    max_items: int | None = None,
    against=None,
    field: str | None = None,
    group_by: str | None = None,
    scorer=None,
    api: str | None = None,
    model: str | None = None,
    budget: float = 0.05,
    concurrency: int = 8,
    batch_size: int = 8,
    timeout: float = 20,
    cache: bool = True,
    scan: str = "shortlist",
    chunk_size: int = 1800,
    overlap: int = 240,
    _transport=None,
) -> Selection:
    """Select evidence. `scorer(task, passages)` may be synchronous or asynchronous.

    `tokens` applies to result.context, not the full JSON metadata or your surrounding prompt.
    `mode=auto` uses Jev if configured, otherwise explicitly reports local lexical selection.
    """
    started = time.perf_counter()
    if not isinstance(task, str) or not task.strip() or len(task) > 4000:
        raise ValueError("task must contain 1..4000 characters")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("tokens must be a nonnegative integer")
    if mode not in {"auto", "local", "semantic"} or scan not in {"shortlist", "all"}:
        raise ValueError("mode must be auto/local/semantic; scan must be shortlist/all")
    if not 0 <= diversity <= 1 or candidates < 1 or (max_items is not None and max_items < 1):
        raise ValueError("diversity must be 0..1; candidates and max_items must be positive")
    if threshold is not None:
        validate_score(threshold)
    # Validate/initialize tokenizer before any model calls or expensive indexing.
    count_tokens("", encoding)
    if tokens == 0:
        return Selection(task, "", [], 0, tokens, encoding, {"mode": mode, "calls": 0, "cost": 0.0})
    backend = None if mode == "local" or scorer else resolve_backend(api, model)
    if mode == "semantic" and not backend and not scorer:
        raise ValueError(
            "semantic mode needs TYPESAFE_API_KEY or OPENROUTER_API_KEY; use --mode local offline"
        )
    mode = "custom" if scorer else "semantic" if backend else "local"
    if scan == "all" and mode == "local":
        raise ValueError("--scan all requires a semantic or custom scorer; local mode uses the lexical index")
    threshold = threshold if threshold is not None else (0.25 if mode != "local" else 0.0)
    own_index = not isinstance(data, Index)
    if own_index and isinstance(data, (str, Path)) and Path(data).suffix == ".jselect":
        index = Index(data)
    else:
        index = (
            Index.build(data, field=field, group_by=group_by, chunk_size=chunk_size, overlap=overlap)
            if own_index
            else data
        )
    judge = None
    warnings = []
    try:
        judge = (
            JevScorer(
                backend,
                budget=budget,
                concurrency=concurrency,
                batch_size=batch_size,
                timeout=timeout,
                cache=cache,
                transport=_transport,
            )
            if backend
            else None
        )
        indexed = time.perf_counter()
        prior = previous_texts(against)
        prior_set = set(prior)
        retrieval_start = time.perf_counter()

        def fitted(data):
            iterator = iter(data)
            while block := list(islice(iterator, 256)):
                yield from (
                    p
                    for p in fit_passages(
                        block, tokens=tokens, encoding=encoding, task=task, local=mode == "local"
                    )
                    if p.text not in prior_set
                )

        async def score_batch(passages):
            if scorer:
                method = scorer.score if hasattr(scorer, "score") else scorer
                scores = method(task, passages)
                if inspect.isawaitable(scores):
                    scores = await scores
                scores = list(scores)
            elif judge:
                scores = await judge.score(task, passages)
            else:
                scores = [p.retrieval_score for p in passages]
            if len(scores) != len(passages):
                raise ValueError("scorer must return exactly one score per passage")
            return [validate_score(s) for s in scores]

        source_considered, scored_count, scoring_seconds = 0, 0, 0.0
        if scan == "all":
            if judge:
                judge.preflight(task, fitted(index.all()))
            passages = []
            stream = iter(fitted(index.all()))
            while block := list(islice(stream, 256)):
                t0 = time.perf_counter()
                block_scores = await score_batch(block)
                scoring_seconds += time.perf_counter() - t0
                scored_count += len(block)
                eligible = [
                    replace(p, retrieval_score=s)
                    for p, s in zip(block, block_scores, strict=True)
                    if s >= threshold and s > 0
                ]
                passages = Index._spread(passages + eligible, candidates)
            scores = [p.retrieval_score for p in passages]
            source_considered = index.stats["unique_passages"]
        else:
            pool = index.candidates(
                task,
                candidates,
                explore=0.2 if mode != "local" else 0,
                exclude={_digest(text) for text in prior},
            )
            source_considered = len(pool)
            passages = list(fitted(pool))
            # Rechunking for very small budgets must not silently increase model work.
            if len(passages) > candidates:
                passages = Index._spread(passages, candidates)
            t0 = time.perf_counter()
            scores = await score_batch(passages)
            scoring_seconds = time.perf_counter() - t0
            scored_count = len(passages)
        retrieved = time.perf_counter()
        if mode == "local":
            warnings.append(
                "Local mode uses lexical relevance and text diversity; no semantic model was called."
            )
        scored = time.perf_counter()
        items = pack(
            passages,
            scores,
            tokens=tokens,
            encoding=encoding,
            diversity=diversity,
            threshold=threshold,
            previous=prior,
            max_items=max_items,
            mode=mode,
        )
        context = render(items)
        if scan != "all" and source_considered < index.stats["unique_passages"]:
            warnings.append(
                "Only shortlisted passages were scored; relevant evidence may exist outside the shortlist."
            )
        if not items:
            warnings.append("No new passage met the relevance threshold and fit the token budget.")
        stats = {
            **index.stats,
            "mode": mode,
            "scan": scan,
            "candidate_limit": candidates,
            "candidates": len(passages),
            "source_passages_considered": source_considered,
            "passages_evaluated": scored_count,
            "selected": len(items),
            "previous_passages": len(prior),
            "calls": 0,
            "cost": 0.0,
            "index_seconds": indexed - started if own_index else 0.0,
            "retrieval_seconds": retrieved - retrieval_start - scoring_seconds,
            "scoring_seconds": scoring_seconds,
            "selection_seconds": time.perf_counter() - scored,
            "seconds": time.perf_counter() - started,
            "selection_method": "greedy_relevance_novelty_per_token",
            "diversity": diversity,
            "threshold": threshold,
        }
        if judge:
            stats.update(judge.stats)
        return Selection(
            task, context, items, count_tokens(context, encoding), tokens, encoding, stats, warnings
        )
    except SemanticError as exc:
        if judge:
            exc.stats = dict(judge.stats)
        raise
    finally:
        if judge:
            judge.close()
        if own_index:
            index.close()


def select(data, *, task: str, tokens: int = 8000, **options) -> Selection:
    """Synchronous entry point. In an async app or notebook, use `await aselect(...)`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aselect(data, task=task, tokens=tokens, **options))
    raise RuntimeError("select() cannot run inside an event loop; use await aselect(...)")
