"""Frozen default-mode outputs: `default_selection.json` was written by 0.1.1, before sampling existed."""

import importlib
import json
from pathlib import Path

from jselect import Index, Record, select

GOLDEN = Path(__file__).with_name("default_selection.json")
TIMINGS = ("index_seconds", "retrieval_seconds", "scoring_seconds", "selection_seconds", "seconds")


def corpus():
    rows = [
        Record(f"Signup email never arrived. The verification message is missing. Attempt {i}.", id=f"e{i}")
        for i in range(12)
    ]
    rows += [Record("The account was locked after a password reset.", id=f"l{i}") for i in range(7)]
    rows += [
        Record("Signup requires a credit card even for the free plan, so I left.", id="card"),
        Record("Signup worked after opening the email link in the original browser.", id="worked"),
        Record("Please send a copy of last month's invoice. " * 40, id="invoice"),
        Record("Can I move my meeting to Friday?", id="calendar"),
    ]
    return rows


def graded(task, passages):
    return [
        0.1 if "invoice" in p.text or "meeting" in p.text else 0.6 + len(p.text) % 7 / 20 for p in passages
    ]


def default_outputs() -> str:
    """Every byte of default-mode JSON, except wall-clock timings and floats beyond 12 decimal places."""
    runs = {
        "custom": select(corpus(), task="signup problems", tokens=900, encoding="bytes", scorer=graded),
        "custom_small": select(
            corpus(), task="signup problems", tokens=260, encoding="bytes", scorer=graded, max_items=2
        ),
        "local": select(corpus(), task="signup email locked", tokens=700, encoding="bytes", mode="local"),
    }
    runs["against"] = select(
        corpus(),
        task="signup problems",
        tokens=900,
        encoding="bytes",
        scorer=graded,
        against=runs["custom_small"],
    )
    # Oversized records whose equal excerpts reach the scan in different blocks of 256, and records
    # whose own metadata uses the key "per": neither may change under sampling or --per support.
    long = "shared boilerplate sentence. " * 50
    rows = [Record(long + f" unique tail {i:04}", id=f"r{i:04}") for i in range(260)]
    fit = importlib.import_module("jselect.select").fit_passages
    with Index.build(rows) as index:
        parts = fit(list(index.all())[:256], tokens=300, encoding="bytes", task="x")
    tails = [p.text for p in parts if "unique tail" in p.text][:50]
    runs["duplicates_across_blocks"] = select(
        rows, task="x", tokens=300, encoding="bytes", against=tails, scorer=lambda t, ps: [0.9] * len(ps)
    )
    runs["metadata_named_per"] = select(
        [
            Record(long + f" unique tail {i:04}", id=f"r{i:04}", metadata={"per": str(i % 2)})
            for i in range(4)
        ],
        task="x",
        tokens=300,
        encoding="bytes",
        scorer=lambda t, ps: [0.1 if "unique tail" in p.text else 0.95 for p in ps],
    )
    payload = {}
    for name, result in runs.items():
        data = result.to_dict()
        data["stats"].update(dict.fromkeys(TIMINGS, 0.0))
        payload[name] = data
    # Python 3.12 changed float summation, which moves the last digit of a novelty score.
    rounded = json.loads(json.dumps(payload), parse_float=lambda text: round(float(text), 12))
    return json.dumps(rounded, ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    GOLDEN.write_text(default_outputs(), encoding="utf-8")
