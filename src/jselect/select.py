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

from jevkit_runtime import resolve

from .index import Index, _digest
from .inputs import records
from .judge import PROVIDERS, JevScorer, SemanticError, validate_score
from .text import chunks, count_tokens, features, similarity, words
from .types import PER, Evidence, Passage, Selection, merge_passages


def render_item(item: Evidence | Passage, number: int, *, sample: bool = False) -> str:
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
    if sample:
        # A content-only citation makes fitting and the draw independent of source locations.
        location = {"passage": item.id}
    draws = getattr(item, "draws", item.occurrences if sample else None)
    if (draws or 1) > 1:
        location["draws"] = draws
    return f"[{number}] {json.dumps(location, ensure_ascii=False, separators=(',', ':'))}\n{item.text}"


def render(items, *, sample: bool = False) -> str:
    return "\n\n".join(render_item(item, i + 1, sample=sample) for i, item in enumerate(items))


def render_sample(items, encoding: str) -> str:
    """Keep location citations when they fit within the content-only citation allowance."""
    canonical = render(items, sample=True)
    located = render(items)
    return located if count_tokens(located, encoding) <= count_tokens(canonical, encoding) else canonical


def previous_texts(against) -> list[str]:
    if against is None:
        return []
    if isinstance(against, Selection):
        return [item.text for item in against.items]
    if isinstance(against, (list, tuple)) and against and all(isinstance(a, Selection) for a in against):
        return [item.text for result in against for item in result.items]
    if isinstance(against, (str, Path)):
        path = Path(against)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "items" in data and "schema_version" in data:
                return [item["text"] for item in data["items"]]
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            # A --per result is one selection per line; ordinary JSONL records fall through.
            lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if lines and all(isinstance(d, dict) and "items" in d and "schema_version" in d for d in lines):
                return [item["text"] for data in lines for item in data["items"]]
        return [rec.text for rec in records(path)]
    if isinstance(against, dict) and "items" in against and "schema_version" in against:
        return [item["text"] for item in against["items"]]
    return [rec.text for rec in records(against)]


def fit_passages(
    passages: list[Passage],
    *,
    tokens: int,
    encoding: str,
    task: str,
    local: bool = False,
    apart=None,
    sample: bool = False,
) -> list[Passage]:
    """Rechunk oversized candidates before judging, so scores describe exactly the returned evidence.

    Equal excerpts are merged; `apart` (see merge_passages) limits that to one --per value or one
    indexed parent, which makes the result independent of how the scan was cut into blocks.
    """
    result = []
    query = set(words(task))
    for p in passages:
        if count_tokens(render_item(p, 1, sample=sample), encoding) <= tokens:
            result.append(p)
            continue
        empty = replace(p, text="") if sample else Passage(p.id, "", p.sources)
        overhead = count_tokens(render_item(empty, 1, sample=sample), encoding)
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
            if count_tokens(render_item(part, 1, sample=sample), encoding) <= tokens:
                result.append(part)
            elif len(part.text) < len(p.text) and len(part.text) > 8:
                result.extend(
                    fit_passages(
                        [part], tokens=tokens, encoding=encoding, task=task, local=local, sample=sample
                    )
                )
    merged = merge_passages(result, apart=apart)
    if sample:
        # Equal fragments within a parent can raise the reserved count after splitting.
        fitted = []
        for p in merged:
            if count_tokens(render_item(p, 1, sample=True), encoding) <= tokens:
                fitted.append(p)
            else:
                fitted.extend(
                    fit_passages(
                        [p],
                        tokens=tokens,
                        encoding=encoding,
                        task=task,
                        local=local,
                        apart=apart,
                        sample=True,
                    )
                )
        return merge_passages(fitted, apart=apart)
    return merged


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


class Pool:
    """The bounded relevance/diversity pool that greedy packing chooses from."""

    def __init__(self, limit: int):
        self.limit, self.pool = limit, []

    def extend(self, passages: list[Passage]) -> None:
        self.pool = Index._spread(self.pool + passages, self.limit)


class Draw:
    """A seeded uniform random order over relevant passage occurrences, taken from the front.

    Every occurrence has an independent uniform key, and the sample is each occurrence whose key
    is below a cutoff: the key of the first passage that does not fit. Identical texts are indexed
    once, so a passage stands in for its occurrences through their smallest key, and the rest of
    its draws are counted afterwards. Only the front of the order is kept in memory; a passage
    behind more than twice the token budget can never be reached.

    A unit is one fitted text within one indexed passage, and the caller adds each unit once. Its
    key depends only on the seed, that parent's ID, and the text, never on caller-supplied record
    IDs or on how the scan was ordered or cut into blocks. Equal texts from different parents are
    separate, independent units that become one item if both are drawn.
    """

    def __init__(self, *, tokens: int, encoding: str, seed: int, max_items: int | None = None):
        self.tokens, self.encoding, self.seed, self.max_items = tokens, encoding, seed, max_items
        # Units at the front of the order: (key, identity) sorted, and identity -> (key, passage, cost).
        self.head: list[tuple] = []
        self.units: dict[tuple, tuple] = {}
        # The smallest key known to lie outside the sample; 1.0 while nothing has been discarded.
        self.bound = 1.0
        self.passages = self.occurrences = 0
        # Set by finish() when the sample is empty although relevant passages exist.
        self.note: str | None = None

    def extend(self, passages: list[Passage]) -> None:
        for passage in passages:
            self.add(passage)

    def add(self, passage: Passage) -> None:
        self.occurrences += passage.occurrences
        parent = passage.sources[0].get("passage_id") if passage.sources else None
        salt = (self.seed, parent or passage.id, passage.id)
        held = self.units.pop(salt, None)
        if held:
            # The same unit again: one unit with the combined count, never two sharing a random number.
            self.head.remove((held[0], salt))
            passage = merge_passages([held[1], passage])[0]
        else:
            self.passages += 1
        # The minimum of n independent uniform keys, computed stably for large n.
        key = -math.expm1(math.log1p(-_uniform(*salt)) / passage.occurrences)
        if key >= self.bound:
            return
        cost = held[2] if held else count_tokens(render_item(passage, 1, sample=True), self.encoding)
        insort(self.head, (key, salt))
        self.units[salt] = (key, passage, cost)
        spent, texts = 0, set()
        for i, (_, unit) in enumerate(self.head):
            _, p, c = self.units[unit]
            spent += 0 if p.text in texts else c
            texts.add(p.text)
            full = self.max_items is not None and len(texts) > self.max_items
            if (full or spent > 2 * self.tokens) and i + 1 < len(self.head):
                self.bound = self.head[i + 1][0]
                for _, dropped in self.head[i + 1 :]:
                    del self.units[dropped]
                del self.head[i + 1 :]
                break

    def finish(self, reason: str) -> tuple[list[Evidence], dict]:
        chosen: dict[str, list] = {}
        cutoff, stop = self.bound, "population" if self.bound == 1.0 else "budget"
        for key, salt in self.head:
            p = self.units[salt][1]
            entry = chosen.get(p.text)
            if entry is None and self.max_items is not None and len(chosen) >= self.max_items:
                cutoff, stop = key, "max_items"
                break
            passage = merge_passages([entry[0], p])[0] if entry else p
            trial = {**chosen, p.text: [passage, [*(entry[1] if entry else []), (key, p.occurrences, salt)]]}
            # Reserve the widest possible draw count so the final header cannot outgrow the budget.
            if (
                count_tokens(render(self._items(trial, None, reason), sample=True), self.encoding)
                > self.tokens
            ):
                cutoff, stop = key, "budget"
                break
            chosen = trial
        while True:
            items = self._items(chosen, cutoff, reason)
            if count_tokens(render(items, sample=True), self.encoding) <= self.tokens:
                break
            # A tokenizer may charge more for a shorter header; end the draw one passage earlier.
            _, (_, parts) = chosen.popitem()
            cutoff, stop = min(key for key, _, _ in parts), "budget"
        if not items and self.passages:
            # Counts can grow when equal fragments from different parents meet in the draw.
            self.note = (
                "The first passage drawn fits the token budget only without the room reserved for its draw "
                "count, and the draw never skips ahead, so the sample is empty. Raise --tokens slightly."
            )
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
    _per: str | None = None,
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
    if _per and scan == "shortlist":
        raise ValueError("--per scans the collection once for every group; omit --scan shortlist")
    if threshold is not None:
        validate_score(threshold)
    # Validate/initialize tokenizer before any model calls or expensive indexing.
    count_tokens("", encoding)
    if tokens == 0 and not _per:
        empty = Selection(task, "", [], 0, tokens, encoding, {"mode": mode, "calls": 0, "cost": 0.0})
        return empty
    backend = (
        None
        if tokens == 0 or mode == "local" or scorer
        else resolve(PROVIDERS, api, model=model, missing_ok=True)
    )
    if tokens and mode == "semantic" and not backend and not scorer:
        raise ValueError(
            "semantic mode needs TYPESAFE_API_KEY or OPENROUTER_API_KEY; use --mode local offline"
        )
    mode = "custom" if scorer else "semantic" if backend else "local"
    if scan is None:
        scan = "shortlist" if mode == "local" and not (sample or _per) else "all"
    if scan == "all" and mode == "local" and not (sample or _per):
        raise ValueError("--scan all requires a semantic or custom scorer; local mode uses the lexical index")
    if threshold is None:
        threshold = 0.0 if mode == "local" else 0.5 if sample else 0.25
    own_index = not isinstance(data, Index)
    if own_index and isinstance(data, (str, Path)) and Path(data).suffix == ".jselect":
        index = Index(data)
    else:
        index = (
            Index.build(
                data, field=field, group_by=group_by, per=_per, chunk_size=chunk_size, overlap=overlap
            )
            if own_index
            else data
        )
    judge = None
    warnings = []
    try:
        if _per and index.stats.get("per") != _per:
            raise ValueError(f"index was not built with --per {_per}; rebuild it with index ... --per {_per}")
        if tokens == 0:
            return [
                Selection(
                    task,
                    "",
                    [],
                    0,
                    tokens,
                    encoding,
                    {**index.stats, "mode": mode, "calls": 0, "cost": 0.0, "passages_evaluated": 0},
                    ["The token budget is zero; no passages were scored."],
                    per={"field": _per, "value": value, **counts},
                )
                for value, counts in index.per_values().items()
            ]
        judge = (
            JevScorer(
                backend,
                budget=budget,
                concurrency=concurrency,
                batch_size=batch_size,
                timeout=timeout,
                cache=cache,
                transport=_transport,
                full_population=bool(sample or _per),
            )
            if backend
            else None
        )
        indexed = time.perf_counter()
        prior = previous_texts(against)
        prior_set = set(prior)
        retrieval_start = time.perf_counter()

        def apart(p):
            ref = p.sources[0]
            # Equal excerpts stay apart across --per values, and across indexed parents when sampling,
            # so a sampled unit never depends on how the scan was cut into blocks.
            return (ref.get(PER) if _per else None, ref.get("passage_id") if sample else None)

        def fitted(data):
            iterator = iter(data)
            while block := list(islice(iterator, 256)):
                yield from (
                    p
                    for p in fit_passages(
                        block,
                        tokens=tokens,
                        encoding=encoding,
                        task=task,
                        local=mode == "local",
                        apart=apart if _per or sample else None,
                        sample=bool(sample),
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
        # One collector per output context: a single one, or one for each --per value.
        parts = index.per_values() if _per else {None: {}}
        groups = {
            value: Draw(tokens=tokens, encoding=encoding, seed=seed, max_items=max_items)
            if sample
            else Pool(candidates)
            for value in parts
        }
        if scan == "all":

            def found():
                # A local population is every lexical match, not the diversified shortlist.
                return fitted(
                    index.matches(task, per=bool(_per)) if mode == "local" else index.all(per=bool(_per))
                )

            def distinct(passages):
                seen = set()
                for p in passages:
                    if p.id not in seen:
                        seen.add(p.id)
                        yield p

            # Sampling and --per score each distinct text once in a scan, however often it recurs across
            # parents or groups; this map is separate from the persistent cache. Ordinary selection keeps
            # the earlier behavior and scores every fitted passage as it arrives.
            known = {} if sample or _per else None
            if judge:
                judge.preflight(task, found() if known is None else distinct(found()))
            # The sampled population is relevance at or above the threshold, zero included. Local mode
            # also needs a shared task term, and the default rule keeps its earlier positive-score floor.
            inclusive = bool(sample) and mode != "local"
            stream = found()
            while True:
                block, fresh, waiting = [], [], set()
                for p in stream:
                    block.append(p)
                    if known is None or (p.id not in known and p.id not in waiting):
                        fresh.append(p)
                        waiting.add(p.id)
                    # A scorer sees at most 256 passages; repeats of known texts ride along unscored.
                    if len(fresh) == 256 or len(block) == 4096:
                        break
                if not block:
                    break
                t0 = time.perf_counter()
                fresh_scores = await score_batch(fresh) if fresh else []
                scoring_seconds += time.perf_counter() - t0
                scored_count += len(fresh)
                if known is None:
                    block_scores = fresh_scores
                else:
                    known.update(zip((p.id for p in fresh), fresh_scores, strict=True))
                    block_scores = [known[p.id] for p in block]
                eligible = {}
                for p, s in zip(block, block_scores, strict=True):
                    if s >= threshold and (s > 0 or inclusive):
                        value = p.sources[0][PER] if _per else None
                        eligible.setdefault(value, []).append(replace(p, retrieval_score=s))
                for value, passages in eligible.items():
                    groups[value].extend(passages)
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
            groups[None].pool = [replace(p, retrieval_score=s) for p, s in zip(passages, scores, strict=True)]
        retrieved = time.perf_counter()
        if mode == "local":
            warnings.append(
                "Local mode samples passages that share a task term; no semantic model judged relevance."
                if sample
                else "Local mode uses lexical relevance and text diversity; no semantic model was called."
            )
        if scan != "all" and source_considered < index.stats["unique_passages"]:
            warnings.append(
                "Only shortlisted passages were scored; relevant evidence may exist outside the shortlist."
            )
        results = []
        for value, group in groups.items():
            scored = time.perf_counter()
            if sample:
                items, sampling = group.finish(
                    f"{mode} relevance at or above {threshold:g}; seeded random draw of passage "
                    "occurrences, without replacement"
                )
            else:
                items = pack(
                    group.pool,
                    [p.retrieval_score for p in group.pool],
                    tokens=tokens,
                    encoding=encoding,
                    diversity=diversity,
                    threshold=threshold,
                    previous=prior,
                    max_items=max_items,
                    mode=mode,
                )
            context = render_sample(items, encoding) if sample else render(items)
            notes = list(warnings)
            if not items:
                notes.append(
                    (sample and group.note)
                    or "No new passage met the relevance threshold and fit the token budget."
                )
            stats = {
                **index.stats,
                "mode": mode,
                "scan": scan,
                "candidate_limit": candidates,
                "candidates": group.passages if sample else len(group.pool),
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
            if sample:
                # Every relevant passage is a candidate for the draw; no pool cap or novelty applies.
                del stats["candidate_limit"], stats["diversity"]
                stats.update(
                    selection_method="random_occurrence_sample_without_replacement",
                    sample=sample,
                    seed=seed,
                    **sampling,
                )
            if judge:
                stats.update(judge.stats)
            results.append(
                Selection(
                    task,
                    context,
                    items,
                    count_tokens(context, encoding),
                    tokens,
                    encoding,
                    stats,
                    notes,
                    per={"field": _per, "value": value, **parts[value]} if _per else None,
                )
            )
        return results if _per else results[0]
    except SemanticError as exc:
        if judge:
            exc.stats = dict(judge.stats)
        raise
    finally:
        if judge:
            await judge.close()
        if own_index:
            index.close()


async def aselect_per(data, *, task: str, per: str, tokens: int = 8000, **options) -> list[Selection]:
    """One context for each value of the `per` field, in order of first appearance.

    The collection is scanned and scored once; selection then runs separately for each value with
    its own `tokens` budget. Other options are those of `aselect`. A saved or reused index must
    have been built with the same `per` field. A `Record` supplies the field in its metadata.
    Values are compared as text. Each result's `per` names its value.
    """
    if not isinstance(per, str) or not per:
        raise ValueError("per must name a record field")
    return await aselect(data, task=task, tokens=tokens, _per=per, **options)


def _blocking(call, name: str):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call())
    raise RuntimeError(f"{name}() cannot run inside an event loop; use await a{name}(...)")


def select(data, *, task: str, tokens: int = 8000, **options) -> Selection:
    """Synchronous entry point. In an async app or notebook, use `await aselect(...)`."""
    return _blocking(lambda: aselect(data, task=task, tokens=tokens, **options), "select")


def select_per(data, *, task: str, per: str, tokens: int = 8000, **options) -> list[Selection]:
    """Synchronous `aselect_per`: one context for each value of the `per` field."""
    return _blocking(lambda: aselect_per(data, task=task, per=per, tokens=tokens, **options), "select_per")
