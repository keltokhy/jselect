"""Evaluate fixed-seed BEIR SciFact queries at equal context budgets; no LLM labels.

Run: uv run python bench/retrieval.py --queries 30 --mode semantic
The script downloads public data, then makes bounded paid requests only in semantic mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
import zipfile
from pathlib import Path

import httpx

from jselect import Index, Record, count_tokens, select
from jselect.judge import PROMPT_VERSION, SemanticError
from jselect.select import render

ROOT = Path(__file__).resolve().parent
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def dataset():
    root = ROOT / "data"
    root.mkdir(exist_ok=True)
    archive = root / "scifact.zip"
    if not archive.exists():
        response = httpx.get(URL, follow_redirects=True, timeout=60)
        response.raise_for_status()
        archive.write_bytes(response.content)
    destination = root / "scifact"
    if not destination.exists():
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not (root / info.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError("unsafe archive member")
            z.extractall(root)
    return destination, hashlib.sha256(archive.read_bytes()).hexdigest()


def baseline(index, task, tokens, candidates):
    passages = index.search(task, candidates)
    selected = []
    for p in passages:
        if p.retrieval_score <= 0:
            continue
        if count_tokens(render([*selected, p])) <= tokens:
            selected.append(p)
    return selected


def ids(items):
    return {ref["record_id"] for item in items for ref in item.sources}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=int, default=30)
    p.add_argument("--mode", choices=["local", "semantic"], default="local")
    p.add_argument("--tokens", type=int, default=2000)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--budget", type=float, default=0.15, help="maximum measured spend across this run")
    p.add_argument("--seed", type=int, default=1909)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()
    root, sha = dataset()
    path = root / "corpus.jselect"
    if not path.exists():

        def corpus():
            with (root / "corpus.jsonl").open() as f:
                for line in f:
                    row = json.loads(line)
                    yield Record(row.get("title", "") + "\n" + row["text"], id=row["_id"], source="scifact")

        with Index.build(corpus(), path=path):
            pass
    queries = {
        r["_id"]: r["text"] for r in map(json.loads, (root / "queries.jsonl").read_text().splitlines())
    }
    gold = {}
    with (root / "qrels" / "test.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if float(row["score"]) > 0:
                gold.setdefault(row["query-id"], set()).add(row["corpus-id"])
    sampled = random.Random(args.seed).sample(sorted(gold), min(args.queries, len(gold)))
    results, spent = [], 0.0
    output = ROOT / "out"
    output.mkdir(exist_ok=True)
    cache_label = "cold" if args.no_cache else "cache-enabled"
    destination = output / (
        f"scifact-{args.mode}-q{args.queries}-c{args.candidates}-s{args.seed}"
        f"-{PROMPT_VERSION}-{cache_label}.json"
    )

    def checkpoint(error=None):
        report = {
            "dataset": URL,
            "sha256": sha,
            "settings": vars(args),
            "prompt_version": PROMPT_VERSION,
            "queries": results,
            "error": error,
            "summary": {
                "queries": len(results),
                "cost": spent,
                **{
                    key: statistics.mean(r[key] for r in results) if results else None
                    for key in ("candidate_recall", "baseline_recall", "selection_recall")
                },
                "median_seconds": statistics.median(r["stats"]["seconds"] for r in results)
                if results
                else None,
            },
        }
        destination.write_text(json.dumps(report, indent=2))
        return report

    with Index(path) as index:
        for qid in sampled:
            task = queries[qid]
            expected = gold[qid]
            pool = index.candidates(task, args.candidates)
            base_start = time.perf_counter()
            base = baseline(index, task, args.tokens, args.candidates)
            base_seconds = time.perf_counter() - base_start
            try:
                result = select(
                    index,
                    task=task,
                    tokens=args.tokens,
                    mode=args.mode,
                    candidates=args.candidates,
                    budget=min(0.05, args.budget - spent),
                    cache=not args.no_cache,
                    timeout=40,
                )
            except SemanticError as e:
                spent += getattr(e, "stats", {}).get("cost", 0)
                checkpoint({"query": qid, "message": str(e), "stats": getattr(e, "stats", {})})
                raise
            spent += result.stats["cost"]
            row = {
                "id": qid,
                "task": task,
                "expected_ids": sorted(expected),
                "candidate_recall": len(expected & ids(pool)) / len(expected),
                "baseline_recall": len(expected & ids(base)) / len(expected),
                "selection_recall": len(expected & ids(result.items)) / len(expected),
                "baseline_ids": sorted(ids(base)),
                "selected_ids": sorted(ids(result.items)),
                "baseline_tokens": count_tokens(render(base)),
                "tokens": result.tokens,
                "baseline_seconds": base_seconds,
                "stats": result.stats,
            }
            results.append(row)
            checkpoint()
            print(
                json.dumps(
                    {k: row[k] for k in ("id", "candidate_recall", "baseline_recall", "selection_recall")}
                ),
                flush=True,
            )
    report = checkpoint()
    print(json.dumps({"output": str(destination), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
