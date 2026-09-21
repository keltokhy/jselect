"""Task-aware evidence selection with explicit retrieval and output budgets."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import random
import time
from bisect import insort
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
    if (getattr(item, "draws", None) or 1) > 1:
        location["draws"] = item.draws
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


def _uniform(*parts) -> float:
    """A reproducible uniform number in [0, 1) from the seed and a passage's identity."""
    digest = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def _generator(*parts) -> random.Random:
    return random.Random(int(_uniform(*parts) * 2**53))


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """Successes in n trials, by geometric gaps; random.binomialvariate needs Python 3.12."""
    if n <= 0 or p <= 0:
        return 0
    if p >= 1:
        return n
    count, position, log_q = 0, 0, math.log1p(-p)
    while True:
        position += int(math.log(1 - rng.random()) / log_q) + 1
        if position > n:
            return count
        count += 1


class Draw:
    """A seeded uniform random order over relevant passage occurrences, taken from the front.

    Every occurrence has an independent uniform key, and the sample is each occurrence whose key
    is below a cutoff: the key of the first passage that does not fit. Identical texts are indexed
    once, so a passage stands in for its occurrences through their smallest key, and the rest of
    its draws are counted afterwards. Only the front of the order is kept in memory; a passage
    behind more than twice the token budget can never be reached.
    """

    def __init__(self, *, tokens: int, encoding: str, seed: int, max_items: int | None = None):
        self.tokens, self.encoding, self.seed, self.max_items = tokens, encoding, seed, max_items
        self.head: list[tuple] = []
        # The smallest key known to lie outside the sample; 1.0 while nothing has been discarded.
        self.bound = 1.0
        self.passages = self.occurrences = 0

    def add(self, passage: Passage) -> None:
        self.passages += 1
        self.occurrences += passage.occurrences
        ref = passage.sources[0] if passage.sources else {}
        # The location keeps equal texts that reach the scan separately statistically independent.
        salt = (self.seed, passage.id, ref.get("source"), ref.get("record_id"), ref.get("start"))
        # The minimum of n independent uniform keys, computed stably for large n.
        key = -math.expm1(math.log1p(-_uniform(*salt)) / passage.occurrences)
        if key >= self.bound:
            return
        cost = count_tokens(render_item(passage, 1), self.encoding)
        insort(self.head, (key, self.passages, passage, cost, salt))
        spent, texts = 0, set()
        for i, (_, _, p, c, _) in enumerate(self.head):
            spent += 0 if p.text in texts else c
            texts.add(p.text)
            full = self.max_items is not None and len(texts) > self.max_items
            if (full or spent > 2 * self.tokens) and i + 1 < len(self.head):
                self.bound = self.head[i + 1][0]
                del self.head[i + 1 :]
                break

    def finish(self, reason: str) -> tuple[list[Evidence], dict]:
        chosen: dict[str, list] = {}
        cutoff, stop = self.bound, "population" if self.bound == 1.0 else "budget"
        for key, _, p, _, salt in self.head:
            entry = chosen.get(p.text)
            if entry is None and self.max_items is not None and len(chosen) >= self.max_items:
                cutoff, stop = key, "max_items"
                break
            passage = merge_passages([entry[0], p])[0] if entry else p
            trial = {**chosen, p.text: [passage, [*(entry[1] if entry else []), (key, p.occurrences, salt)]]}
            # Reserve the widest possible draw count so the final header cannot outgrow the budget.
            if count_tokens(render(self._items(trial, None, reason)), self.encoding) > self.tokens:
                cutoff, stop = key, "budget"
                break
            chosen = trial
        while True:
            items = self._items(chosen, cutoff, reason)
            if count_tokens(render(items), self.encoding) <= self.tokens:
                break
            # A tokenizer may charge more for a shorter header; end the draw one passage earlier.
            _, (_, parts) = chosen.popitem()
            cutoff, stop = min(key for key, _, _ in parts), "budget"
        return items, {
            "population_passages": self.passages,
            "population_occurrences": self.occurrences,
            "sample_passages": len(items),
            "sample_occurrences": sum(item.draws for item in items),
            "sample_stop": stop,
        }

    def _items(self, chosen: dict[str, list], cutoff: float | None, reason: str) -> list[Evidence]:
        items = []
        for passage, parts in chosen.values():
            draws = passage.occurrences
            if cutoff is not None:
                # Beyond its smallest key, each other occurrence is uniform on the rest of the interval.
                draws = sum(
                    1 + _binomial(_generator(*salt, "repeats"), n - 1, (cutoff - key) / (1 - key))
                    for key, n, salt in parts
                )
            items.append(
                Evidence(
                    passage.id,
                    passage.text,
                    passage.sources,
                    passage.occurrences,
                    passage.retrieval_score,
                    None,
                    reason,
                    draws,
                )
            )
        return items


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
    scan: str | None = None,
    sample: str | None = None,
    seed: int = 0,
    chunk_size: int = 1800,
    overlap: int = 240,
    _transport=None,
) -> Selection:
    """Select evidence. `scorer(task, passages)` may be synchronous or asynchronous.

    `tokens` applies to result.context, not the full JSON metadata or your surrounding prompt.
    `mode=auto` uses Jev if configured, otherwise explicitly reports local lexical selection.
    Omitting `scan` scores all eligible passages with semantic/custom scorers; local mode uses
    the lexical shortlist. Set `scan="shortlist"` to bound passages evaluated before scoring.
    `sample="representative"` replaces relevance/novelty selection with a seeded random draw from
    every passage at or above `threshold` (default 0.5; any lexical match in local mode).
    """
    started = time.perf_counter()
    if not isinstance(task, str) or not task.strip() or len(task) > 4000:
        raise ValueError("task must contain 1..4000 characters")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("tokens must be a nonnegative integer")
    if mode not in {"auto", "local", "semantic"} or scan not in {None, "shortlist", "all"}:
        raise ValueError("mode must be auto/local/semantic; scan must be shortlist/all")
    if not 0 <= diversity <= 1 or candidates < 1 or (max_items is not None and max_items < 1):
        raise ValueError("diversity must be 0..1; candidates and max_items must be positive")
    if sample not in {None, "representative"}:
        raise ValueError("sample must be representative, or omitted for relevance/novelty selection")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if sample and scan == "shortlist":
        raise ValueError("--sample representative draws from every relevant passage; omit --scan shortlist")
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
    if scan is None:
        scan = "shortlist" if mode == "local" and not sample else "all"
    if scan == "all" and mode == "local" and not sample:
        raise ValueError("--scan all requires a semantic or custom scorer; local mode uses the lexical index")
    if threshold is None:
        threshold = 0.0 if mode == "local" else 0.5 if sample else 0.25
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
        draw = Draw(tokens=tokens, encoding=encoding, seed=seed, max_items=max_items) if sample else None
        if scan == "all":
            if judge:
                judge.preflight(task, fitted(index.all()))
            passages = []
            # A local population is every lexical match, not the diversified shortlist.
            stream = iter(fitted(index.matches(task) if mode == "local" else index.all()))
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
                if draw:
                    for p in eligible:
                        draw.add(p)
                else:
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
                "Local mode samples passages that share a task term; no semantic model judged relevance."
                if draw
                else "Local mode uses lexical relevance and text diversity; no semantic model was called."
            )
        scored = time.perf_counter()
        if draw:
            items, sampling = draw.finish(
                f"{mode} relevance at or above {threshold:g}; seeded random draw of passage occurrences, "
                "without replacement"
            )
        else:
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
        if draw:
            # Every relevant passage is a candidate for the draw; no pool cap or novelty applies.
            del stats["candidate_limit"], stats["diversity"]
            stats.update(
                candidates=draw.passages,
                selection_method="random_occurrence_sample_without_replacement",
                sample=sample,
                seed=seed,
                **sampling,
            )
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
