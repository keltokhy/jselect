"""Regressions from the independent review of representative sampling and --per."""

import importlib
import json
from collections import Counter
from unittest.mock import patch

import httpx
import pytest

from jselect import Index, Passage, Record, count_tokens, select, select_per
from jselect.select import Draw, fit_passages

# The package exports the select function under the same name as its module.
module = importlib.import_module("jselect.select")
BOILERPLATE = "shared boilerplate sentence. " * 50


def high(task, passages):
    return [0.9] * len(passages)


def summary(result):
    keys = ("population_passages", "population_occurrences", "sample_occurrences", "sample_stop")
    items = [(item.text, item.occurrences, item.draws) for item in result.items]
    return result.context, items, [result.stats.get(key) for key in keys]


def test_fragments_of_two_records_sharing_a_user_id_are_drawn_independently():
    # Two oversized records reuse id="dup" and sit in different fitting blocks of 256.
    rows = [Record(BOILERPLATE + " unique tail first", id="dup")]
    rows += [Record(f"irrelevant filler {i}", id=f"f{i}") for i in range(255)]
    rows += [Record(BOILERPLATE + " unique tail second", id="dup"), Record("B", id="B")]

    def relevant(task, passages):
        return [
            0.9 if p.text == "B" or (p.text.startswith("shared") and "tail" not in p.text) else 0
            for p in passages
        ]

    wins = Counter()
    with Index.build(rows) as index:
        for seed in range(500):
            result = select(
                index,
                task="x",
                tokens=300,
                encoding="bytes",
                sample="representative",
                seed=seed,
                scorer=relevant,
                max_items=1,
            )
            assert result.stats["population_occurrences"] == 25
            wins[result.items[0].text == "B"] += 1
    # 24 of the 25 relevant occurrences are the shared excerpt.
    assert 0.945 < wins[False] / 500 < 0.975


def test_equal_fragments_cut_from_different_passages_have_independent_keys():
    def fragment(parent, line):
        ref = {
            "source": "same.jsonl",
            "record_id": "dup",
            "start": 0,
            "end": 1,
            "line": line,
            "passage_id": parent,
        }
        return Passage("A", "A", [ref], 1, 0.9)

    other = Passage(
        "B",
        "B",
        [{"source": "same.jsonl", "record_id": "B", "start": 0, "end": 1, "line": 3, "passage_id": "B"}],
        1,
        0.9,
    )
    outcomes = Counter()
    for seed in range(3000):
        draw = Draw(tokens=10_000, encoding="bytes", seed=seed, max_items=1)
        draw.extend([fragment("first-parent", 1), fragment("second-parent", 2), other])
        items, _ = draw.finish("rule")
        outcomes[items[0].text, items[0].draws] += 1
    assert 0.64 < (outcomes["A", 1] + outcomes["A", 2]) / 3000 < 0.70
    assert 0.30 < outcomes["A", 1] / 3000 < 0.37 and 0.30 < outcomes["A", 2] / 3000 < 0.37


def test_sample_does_not_depend_on_scan_order_or_fitting_block_size():
    rows = [Record(BOILERPLATE + f" unique tail {i:04}", id=f"r{i:04}") for i in range(260)]
    with Index.build(rows) as index:
        saved = list(index.all())
        real = module.islice
        for seed in range(6):
            options = {
                "task": "x",
                "tokens": 300,
                "encoding": "bytes",
                "sample": "representative",
                "scorer": high,
            }
            normal = summary(select(index, seed=seed, **options))
            with patch.object(index, "all", lambda **kw: iter(reversed(saved))):
                assert summary(select(index, seed=seed, **options)) == normal
            with patch.object(module, "islice", lambda it, n: real(it, 128 if n == 256 else n)):
                assert summary(select(index, seed=seed, **options)) == normal


def test_per_group_sample_equals_the_group_selected_alone_when_texts_are_shared():
    rows = [
        Record(BOILERPLATE + f" unique tail {i:04}", id=f"r{i:04}", metadata={"g": group})
        for i in range(130)
        for group in ("A", "B")
    ]
    options = {"task": "x", "tokens": 300, "encoding": "bytes", "sample": "representative", "scorer": high}
    with (
        Index.build(rows, per="g") as index,
        Index.build([r for r in rows if r.metadata["g"] == "A"]) as alone,
    ):
        for seed in range(20):
            together = select_per(index, per="g", seed=seed, **options)[0]
            assert summary(together) == summary(select(alone, seed=seed, **options))


def test_an_ordinary_query_on_a_per_index_counts_every_occurrence():
    rows = [
        {"id": f"r{i:04}", "g": str(i % 2), "text": BOILERPLATE + f" unique tail {i:04}"} for i in range(4)
    ]
    with Index.build(rows, per="g") as index, Index.build(rows) as plain:
        fitted = fit_passages(list(plain.all()), tokens=300, encoding="bytes", task="x")
        against = [p.text for p in fitted if "unique tail" in p.text]
        for sample in (None, "representative"):
            options = {"task": "x", "tokens": 300, "encoding": "bytes", "scorer": high, "against": against}
            result, expected = (select(i, sample=sample, **options) for i in (index, plain))
            assert [(i.text, i.occurrences, i.draws) for i in result.items] == [
                (i.text, i.occurrences, i.draws) for i in expected.items
            ]
            assert result.items[0].occurrences == 48 and result.context == expected.context
            if sample:
                assert result.items[0].draws == 48 and result.stats["population_occurrences"] == 48


def test_per_scores_a_shared_excerpt_once_and_pays_for_it_once(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rows = [{"id": f"r{i:04}", "g": f"G{i:04}", "text": BOILERPLATE + " unique tail"} for i in range(300)]
    seen = []

    def scorer(task, passages):
        seen.extend(p.text for p in passages)
        return [0.9] * len(passages)

    assert len(select_per(rows, task="x", per="g", tokens=300, encoding="bytes", scorer=scorer)) == 300
    assert sorted(Counter(seen).values()) == [1, 1]
    estimates = []
    for per in (False, True):
        calls = []

        async def fake(request, calls=calls):
            body = json.loads(request.content)
            calls.append(body)
            answers = {key: {"noul": 0.9} for key in body["state"]}
            return httpx.Response(
                200, json={"answers": answers, "usage": {"cost": 0.000001, "input_tokens": 1}}
            )

        options = {"task": "x", "tokens": 300, "encoding": "bytes", "mode": "semantic", "cache": False}
        options["_transport"] = httpx.MockTransport(fake)
        results = select_per(rows, per="g", **options) if per else [select(rows, **options)]
        assert (len(calls), sum(len(call["state"]) for call in calls)) == (1, 2)
        estimates.append(results[0].stats["estimated_cost_upper_bound"])
    assert estimates[0] == estimates[1]


def test_per_values_are_text_in_both_input_paths():
    mixed = [{"g": 1, "text": "integer group"}, {"g": "1", "text": "string group"}]
    assert [r.per["value"] for r in select_per(mixed, task="x", per="g", scorer=high)] == ["1"]
    records = [Record("hello", id="a", metadata={"g": 1}), Record("again", id="b", metadata={"g": "1"})]
    results = select_per(records, task="x", per="g", scorer=high)
    assert [(r.per["value"], len(r.items)) for r in results] == [("1", 2)]
    for bad in (True, None, ["a"], 1.5):
        with pytest.raises(ValueError, match="--per"):
            select_per([Record("hello", metadata={"g": bad})], task="x", per="g", scorer=high)
        with pytest.raises(ValueError, match="--per"):
            select_per([{"g": bad, "text": "hello"}], task="x", per="g", scorer=high)


def test_per_lines_follow_first_appearance_and_keep_groups_without_passages():
    rows = [{"g": "z", "text": "same"}, {"g": "blank", "text": "   "}, {"g": "a", "text": "same"}]
    results = select_per(rows, task="x", per="g", scorer=high, encoding="bytes")
    assert [r.per["value"] for r in results] == ["z", "blank", "a"]
    blank = results[1]
    assert (blank.per["passages"], blank.per["occurrences"], blank.context) == (0, 0, "")
    with Index.build(rows, per="g") as index:
        assert index.stats["per_values"] == 3


def test_threshold_zero_is_inclusive_except_for_the_local_term_match():
    def scorer(task, passages):
        return [float(p.text == "positive") for p in passages]

    rows = [Record("zero"), Record("positive")]
    result = select(rows, task="x", scorer=scorer, threshold=0, encoding="bytes", sample="representative")
    assert result.stats["population_passages"] == 2 and len(result.items) == 2
    # The default rule keeps its positive-score floor, as in earlier releases.
    assert len(select(rows, task="x", scorer=scorer, threshold=0, encoding="bytes").items) == 1
    local = select(["battery fault", "garden flowers"], task="battery", mode="local", sample="representative")
    assert local.stats["population_passages"] == 1


def test_an_empty_sample_says_when_the_reserved_repeat_count_is_the_reason():
    text = "A" + "x" * 10
    ref = {"source": "toy", "line": 1, "record_id": "A", "start": 0, "end": len(text), "passage_id": "A"}
    passage = Passage("A", text, [ref], 3, 0.9)
    # Fitting now reserves draws first. The collector still diagnoses unfitted input or a reserve
    # that grows when equal fragments from different parents are combined.
    bare = count_tokens(module.render([Passage("A", text, [ref])], sample=True), "bytes")
    tight = Draw(tokens=bare, encoding="bytes", seed=0)
    tight.add(passage)
    items, stats = tight.finish("rule")
    assert items == [] and stats["population_occurrences"] == 3
    assert "room reserved for its draw count" in tight.note
    roomy = Draw(tokens=bare + 10, encoding="bytes", seed=0)
    roomy.add(passage)
    items, _ = roomy.finish("rule")
    assert items[0].draws == 3 and roomy.note is None
