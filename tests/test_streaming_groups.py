import asyncio
import json

import httpx
import pytest

from jselect import Index, Passage, Record, select
from jselect.inputs import read_paths, records
from jselect.judge import Backend, JevScorer, SemanticError


def test_grouping_nonadjacent_rows_preserves_order_and_sources(tmp_path):
    rows = [
        {"thread": "a", "role": "user", "content": "Please stop billing me."},
        {"thread": "b", "role": "user", "content": "Where is my invoice?"},
        {"thread": "a", "role": "assistant", "content": "Your subscription is now canceled."},
    ]
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    groups = list(read_paths([path], group_by="thread"))
    assert [g.id for g in groups] == ["a", "b"]
    assert groups[0].text.index("Please stop") < groups[0].text.index("now canceled")
    assert [m["line"] for m in groups[0].metadata["members"]] == [1, 3]
    result = select(path, task="subscription canceled", group_by="thread", mode="local")
    assert result.items[0].sources[0]["group_id"] == "a"
    for member in result.items[0].sources[0]["members"]:
        original = json.dumps(rows[member["line"] - 1], ensure_ascii=False, sort_keys=True)
        assert groups[0].text[member["start"] : member["end"]] == original


def test_grouping_python_records_and_selected_field():
    data = [{"id": "t1", "session": 1, "message": "alpha"}, {"id": "t2", "session": 1, "message": "beta"}]
    group = list(records(data, group_by="session", field="message"))[0]
    assert group.text == "alpha\n\nbeta"
    assert [m["record_id"] for m in group.metadata["members"]] == ["t1", "t2"]
    result = select(data, task="alpha", group_by="session", field="message", mode="local")
    assert result.items[0].text == group.text


@pytest.mark.parametrize("data", [[{"text": "missing"}], ["plain"], [{"thread": [], "text": "bad"}]])
def test_invalid_group_keys_are_errors(data):
    with pytest.raises(ValueError):
        select(data, task="test", group_by="thread", mode="local")


@pytest.mark.parametrize("options", [{}, {"scan": "all"}], ids=["default", "explicit"])
def test_full_scan_batches_work_and_retains_late_evidence(options):
    calls = []

    def scorer(task, passages):
        calls.append(len(passages))
        return [0.99 if "needle" in p.text else 0.01 for p in passages]

    result = select(
        (f"Record {i}: " + ("needle" if i == 899 else "noise") for i in range(900)),
        task="look for hidden evidence",
        scorer=scorer,
        candidates=32,
        **options,
    )
    assert len(result.items) == 1 and "899" in result.items[0].text
    assert max(calls) <= 256 and sum(calls) == 900
    assert result.stats["passages_evaluated"] == 900
    assert result.stats["candidates"] <= 32
    assert result.stats["scan"] == "all"
    assert result.stats["source_passages_considered"] == 900


@pytest.mark.parametrize("options", [{}, {"scan": "all"}], ids=["default", "explicit"])
def test_full_scan_preflight_does_not_spend_part_of_unaffordable_scan(monkeypatch, options):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")

    def forbidden(request):
        raise AssertionError("network request before complete budget check")

    with pytest.raises(SemanticError, match="budget"):
        select(
            (f"long record {i} " * 40 for i in range(1000)),
            task="records",
            mode="semantic",
            budget=0.001,
            _transport=httpx.MockTransport(forbidden),
            **options,
        )


def test_rechunking_does_not_exceed_shortlist_bound():
    evaluated = []

    def scorer(task, passages):
        evaluated.append(len(passages))
        return [1] * len(passages)

    select(
        ["large document. " * 100],
        task="large document",
        tokens=100,
        candidates=2,
        scorer=scorer,
        scan="shortlist",
    )
    assert max(evaluated) <= 2


def test_batch_context_limits_for_unicode():
    scorer = JevScorer(Backend("fixture", "https://example.test", "fixture", "key", "env"), cache=False)
    passages = [(i, Passage(str(i), "🌈" * 1600, []), "key") for i in range(16)]
    planned = list(scorer.plans("🌈" * 3000, passages))
    assert len(planned) > 2
    assert sum(len(batch) for batch, body in planned) == len(passages)
    for _, body in planned:
        assert len(json.dumps(body, ensure_ascii=False).encode()) < 60000


def test_metadata_cannot_overwrite_verified_provenance():
    with Index.build([Record("refund", source="real", metadata={"source": "fake", "start": 999})]) as index:
        ref = next(index.all()).sources[0]
    assert ref["source"] == "real" and ref["start"] == 0


def test_provider_total_timeout_and_error_usage(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")

    async def slow(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"answers": {"p0": {"noul": 1}}})

    with pytest.raises(SemanticError, match="TimeoutError") as caught:
        select(
            ["some passage"], task="test", mode="semantic", timeout=0.01, _transport=httpx.MockTransport(slow)
        )
    assert caught.value.stats["calls"] == 0


def test_fitted_excerpts_can_be_traced_back_to_saved_parent(tmp_path):
    path = tmp_path / "index.jselect"
    original = "Database retries should use exponential backoff. " * 40
    with Index.build([original], path=path) as index:
        result = select(index, task="database retries", mode="local", tokens=180)
        assert result.items
        for item in result.items:
            ref = item.sources[0]
            parent = index.get(ref["passage_id"])
            offset = parent.sources[0]["start"]
            assert parent.text[ref["start"] - offset : ref["end"] - offset] == item.text


@pytest.mark.parametrize("usage", [{"input_tokens": "bad"}, {"input_tokens": -1}, {"cost": "bad"}])
def test_invalid_usage_is_a_structured_provider_error(monkeypatch, usage):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"answers": {"p0": {"noul": 1}}, "usage": usage})
    )
    with pytest.raises(SemanticError, match="usage"):
        select(["text"], task="test", mode="semantic", _transport=transport)


def test_explicit_binary_input_is_an_error(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello\0world")
    with pytest.raises(ValueError, match="binary input"):
        select(path, task="hello", mode="local")


def test_provider_retry_is_bounded_and_accounted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200, json={"answers": {"p0": {"noul": 0.9}}, "usage": {"input_tokens": 100, "cost": None}}
        )

    result = select(["text"], task="test", mode="semantic", _transport=httpx.MockTransport(handler))
    assert len(calls) == 2 and result.stats["retries"] == 1
    assert result.stats["cost"] > 0 and result.stats["cost_source"] == "estimated_at_list_price"


def test_dense_unicode_is_rechunked_to_fit_small_token_budget():
    original = "🌈" * 1000
    result = select([original], task="rainbows", tokens=150, scorer=lambda q, ps: [0.9] * len(ps))
    assert result.items and result.tokens <= 150
    for p in result.items:
        ref = p.sources[0]
        assert p.text == original[ref["start"] : ref["end"]]


def test_identical_fitted_passages_do_not_repeat_and_keep_source_references():
    from jselect.types import merge_passages

    merged = merge_passages(
        [Passage("same", "text", [{"source": "a"}]), Passage("same", "text", [{"source": "b"}])]
    )
    assert len(merged) == 1 and merged[0].sources == [{"source": "a"}, {"source": "b"}]
    from fractions import Fraction

    result = select(["one"], task="test", scorer=lambda q, p: [Fraction(1, 2)])
    assert result.items[0].relevance == 0.5
