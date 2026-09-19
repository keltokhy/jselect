import asyncio
import hashlib
import json

import pytest

from jselect import Index, Record, aselect, count_tokens, select
from jselect.inputs import read_paths


def all_relevant(task, passages):
    return [0.9] * len(passages)


@pytest.mark.parametrize("encoding", ["o200k_base", "cl100k_base", "bytes"])
@pytest.mark.parametrize("budget", [0, 1, 80, 150, 400, 1600])
def test_exact_budget_and_unicode_source_integrity(encoding, budget):
    original = "Payment café 東京 👩🏽‍💻 — <|endoftext|>\n" * 160 + "Refund arrived at last."
    result = select(
        [Record(original, id="unicode", source="a.txt")],
        task="payment refund",
        tokens=budget,
        encoding=encoding,
        scorer=all_relevant,
        chunk_size=200,
        overlap=30,
    )
    assert result.tokens == count_tokens(result.context, encoding) <= budget
    for item in result.items:
        ref = item.sources[0]
        assert item.text == original[ref["start"] : ref["end"]]
        assert ref["record_sha256"] == hashlib.sha256(original.encode()).hexdigest()
        assert ref["line"] == 1 + original[: ref["start"]].count("\n")
    if budget >= 150:
        assert result.items


def test_long_document_tail_is_searched(tmp_path):
    path = tmp_path / "manual.txt"
    path.write_text("Unrelated material.\n" * 1000 + "To disable retries, set max_attempts to one.")
    result = select(path, task="disable retries", mode="local", tokens=300)
    assert "max_attempts" in result.context
    ref = result.items[0].sources[0]
    assert ref["start"] > 18000
    assert result.items[0].text == path.read_text()[ref["start"] : ref["end"]]


def test_duplicate_sources_are_retained():
    values = [Record("The account was locked.", id=str(i), source=f"file{i}.txt") for i in range(9)]
    with Index.build(values) as index:
        assert index.stats["records"] == 9
        assert index.stats["unique_passages"] == 1
        result = select(index, task="account locked", mode="local")
    assert len(result.items) == 1
    assert result.items[0].occurrences == 9
    assert len(result.items[0].sources) == 5


def test_diversity_preserves_a_distinct_issue_and_opposite_evidence():
    texts = [
        f"Signup email never arrived. The verification message is missing. Attempt {i}." for i in range(20)
    ]
    texts += [
        "Signup requires a credit card even for the free plan, so I left.",
        "Signup worked after opening the email link in the original browser.",
    ]
    result = select(texts, task="signup problems", scorer=all_relevant, tokens=1500, max_items=3)
    assert any("credit card" in p.text for p in result.items)
    assert any("worked" in p.text for p in result.items)


def test_against_seeks_fresh_evidence_beyond_original_shortlist(tmp_path):
    values = [f"Battery fault observed on device {i}." for i in range(30)]
    with Index.build(values) as index:
        first = select(index, task="battery fault", mode="local", candidates=3, max_items=3)
        second = select(index, task="battery fault", mode="local", candidates=3, against=first)
        saved = tmp_path / "selection.json"
        saved.write_text(json.dumps(first.to_dict()))
        replay = select(index, task="battery fault", mode="local", candidates=3, against=saved)
    assert len(first.items) == len(second.items) == 3
    assert not {p.id for p in first.items} & {p.id for p in second.items}
    assert [p.id for p in replay.items] == [p.id for p in second.items]


def test_empty_and_irrelevant_data_produce_empty_context():
    for data in ([], [""], ["Garden flowers blooming"]):
        result = select(data, task="database deadlock", mode="local")
        assert result.context == "" and result.tokens == 0 and result.items == []


@pytest.mark.parametrize("scores", [[float("nan")], [2], [True], [], [0.3, 0.4]])
def test_custom_scorer_must_return_valid_aligned_scores(scores):
    with pytest.raises((ValueError, RuntimeError)):
        select(["evidence"], task="question", scorer=lambda q, p: scores)


def test_async_scorer_and_async_api():
    async def scorer(task, passages):
        await asyncio.sleep(0)
        return [0.9] * len(passages)

    async def run():
        with pytest.raises(RuntimeError, match="aselect"):
            select(["text"], task="test")
        return await aselect(["text"], task="test", scorer=scorer)

    assert asyncio.run(run()).items[0].text == "text"


def test_zero_budget_never_reads_input_or_calls_model():
    def forbidden():
        raise AssertionError("read input")
        yield

    assert select(forbidden(), task="test", tokens=0).tokens == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokens": -1},
        {"tokens": True},
        {"candidates": 0},
        {"diversity": float("nan")},
        {"threshold": 2},
        {"max_items": 0},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        select(["text"], task="test", **kwargs)


def test_index_is_reusable_and_failed_rebuild_is_atomic(tmp_path):
    path = tmp_path / "saved.jselect"
    with Index.build(["database deadlock"], path=path) as index:
        identifier = next(index.all()).id
    original = path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        Index.build(["other"], path=path)
    with pytest.raises(ValueError):
        Index.build([42], path=path, force=True)
    assert path.read_bytes() == original
    with Index(path) as index:
        assert index.get(identifier).text == "database deadlock"
        assert index.select(task="deadlock", mode="local").items
    assert select(path, task="deadlock", mode="local").items[0].text == "database deadlock"


def test_input_adapters_keep_locations_and_ids(tmp_path):
    csv = tmp_path / "tickets.csv"
    csv.write_text('id,body\na,"line 1\nline 2"\nb,next\n')
    rows = list(read_paths([csv]))
    assert [(r.id, r.line, r.text) for r in rows] == [("a", 2, "line 1\nline 2"), ("b", 4, "next")]
    data = tmp_path / "data.json"
    data.write_text('[{"message":"first"},{"message":"second"}]')
    rows = list(read_paths([data]))
    assert [r.id for r in rows] == ["1", "2"]
    assert [r.metadata for r in rows] == [{"json_index": 0}, {"json_index": 1}]
    nested = tmp_path / "nested.jsonl"
    nested.write_text('{"id":8,"event":{"description":"a refund"}}\n')
    assert list(read_paths([nested], field="event.description"))[0].text == "a refund"


def test_directory_discovery_respects_ignores_and_skips_binary_secrets(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("private text")
    (tmp_path / ".env").write_text("API_KEY=secret")
    (tmp_path / "image.bin").write_bytes(b"image\0content")
    (tmp_path / "useful.txt").write_text("refund instructions")
    result = select(tmp_path, task="refund", mode="local")
    assert len(result.items) == 1
    assert result.items[0].text == "refund instructions"


def test_malformed_structured_input_is_explicit(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"fine"}\n{bad}\n')
    with pytest.raises(ValueError, match="invalid jsonl"):
        select(path, task="fine", mode="local")
