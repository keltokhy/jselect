import asyncio
import io
import json
from collections import Counter

import httpx
import pytest

from jselect import Index, Record, aselect_per, count_tokens, select, select_per
from jselect.cli import main
from jselect.types import PER


def rows():
    data = []
    for company, size in (("acme-2021", 60), ("bolt-2021", 6), ("cask-2021", 3)):
        data += [
            {"id": f"{company}-{i}", "company_year": company, "text": f"strong {company} complaint {i}."}
            for i in range(size)
        ]
        data += [
            {"id": f"{company}-t{i}", "company_year": company, "text": "strong template: account locked."}
            for i in range(size // 3)
        ]
    data.append({"id": "dune-1", "company_year": "dune-2021", "text": "unrelated note about lunch."})
    return data


class Scorer:
    def __init__(self):
        self.texts = []

    def score(self, task, passages):
        self.texts += [p.text for p in passages]
        return [0.9 if "strong" in p.text else 0.05 for p in passages]


OPTIONS = {"task": "complaints", "per": "company_year", "encoding": "bytes", "tokens": 700}


@pytest.mark.parametrize("sample", [None, "representative"])
def test_each_group_gets_its_own_budgeted_context_from_one_shared_scan(sample):
    scorer = Scorer()
    results = select_per(rows(), scorer=scorer, sample=sample, seed=3, **OPTIONS)
    assert [r.per["value"] for r in results] == ["acme-2021", "bolt-2021", "cask-2021", "dune-2021"]
    # The template occurs in three groups and is still scored exactly once.
    assert Counter(scorer.texts).most_common(1)[0][1] == 1 and len(scorer.texts) == 71
    for result in results:
        assert result.tokens == count_tokens(result.context, "bytes") <= 700
        assert result.stats["passages_evaluated"] == 71
        assert all(ref[PER] == result.per["value"] for item in result.items for ref in item.sources)
    acme, bolt, cask, dune = results
    # A large group neither starves a small one nor lends it a budget.
    assert acme.tokens > 500 and bolt.items and cask.items
    assert dune.items == [] and dune.context == "" and dune.per["passages"] == 1
    assert any("No new passage" in warning for warning in dune.warnings)
    assert (acme.per["passages"], acme.per["occurrences"]) == (61, 80)
    templates = [item for result in results for item in result.items if "template" in item.text]
    assert {item.occurrences for item in templates} <= {20, 2, 1}


@pytest.mark.parametrize("sample", [None, "representative"])
def test_a_group_is_selected_as_if_it_were_the_whole_collection(sample):
    # Records share one line number, so citation headers cost the same with or without the other groups.
    data = [
        Record(row["text"], id=row["id"], metadata={"company_year": row["company_year"]}) for row in rows()
    ]
    options = {k: v for k, v in OPTIONS.items() if k != "per"}
    results = select_per(data, scorer=Scorer(), sample=sample, seed=5, **OPTIONS)
    for result in results:
        alone = [rec for rec in data if rec.metadata["company_year"] == result.per["value"]]
        expected = select(alone, scorer=Scorer(), sample=sample, seed=5, **options)
        assert result.context == expected.context
        assert [item.occurrences for item in result.items] == [item.occurrences for item in expected.items]
        if sample:
            for key in ("population_passages", "population_occurrences", "sample_occurrences", "sample_stop"):
                assert result.stats[key] == expected.stats[key]
    with pytest.raises(ValueError, match="Record metadata needs 'company_year'"):
        select_per([Record("strong text", id="bare")], scorer=Scorer(), **OPTIONS)


def test_representative_counts_are_per_group():
    results = select_per(rows(), scorer=Scorer(), sample="representative", **{**OPTIONS, "tokens": 100_000})
    sizes = {
        r.per["value"]: (r.stats["population_passages"], r.stats["population_occurrences"]) for r in results
    }
    assert sizes == {"acme-2021": (61, 80), "bolt-2021": (7, 8), "cask-2021": (4, 4), "dune-2021": (0, 0)}
    acme = results[0]
    assert acme.stats["sample_occurrences"] == 80 and '"draws":20}' in acme.context


def test_local_groups_are_not_starved_by_a_global_shortlist():
    data = [{"site": "big", "text": f"battery fault on device {i}, battery fault again."} for i in range(300)]
    data += [{"site": "small", "text": "one battery was replaced."}]
    results = select_per(data, task="battery fault", per="site", mode="local", tokens=400, candidates=8)
    assert [(r.per["value"], bool(r.items)) for r in results] == [("big", True), ("small", True)]
    assert all(r.stats["scan"] == "all" and r.stats["mode"] == "local" for r in results)
    with pytest.raises(ValueError, match="omit --scan shortlist"):
        select_per(data, task="battery fault", per="site", mode="local", scan="shortlist")


@pytest.mark.parametrize("sample", [None, "representative"])
def test_budgets_smaller_than_a_passage_split_per_group_and_score_shared_excerpts_once(sample):
    boilerplate = "strong shared boilerplate sentence number one. " * 30
    data = [
        {"id": f"{group}{i}", "group": group, "text": boilerplate + f" strong tail {group if i else ''}"}
        for group in ("alpha", "beta-with-a-longer-name")
        for i in range(3)
    ]
    scorer = Scorer()
    results = select_per(
        data, task="t", per="group", tokens=300, encoding="bytes", scorer=scorer, sample=sample
    )
    assert max(Counter(scorer.texts).values()) == 1
    for result in results:
        assert result.items and result.tokens == count_tokens(result.context, "bytes") <= 300
        for item in result.items:
            ref = item.sources[0]
            original = next(row["text"] for row in data if row["id"] == ref["record_id"])
            assert ref[PER] == result.per["value"] and item.text == original[ref["start"] : ref["end"]]


def test_against_applies_to_every_group(tmp_path):
    first = select_per(rows(), scorer=Scorer(), **OPTIONS)
    seen = [item.text for result in first for item in result.items]
    saved = tmp_path / "dossiers.jsonl"
    saved.write_text("".join(json.dumps(result.to_dict()) + "\n" for result in first))
    # Earlier per-group results are accepted as returned, as saved JSON Lines, or as plain texts.
    for against in (first, saved, seen):
        again = select_per(rows(), scorer=Scorer(), against=against, **OPTIONS)
        assert again[0].items and not set(seen) & {i.text for result in again for i in result.items}
        assert all(result.stats["previous_passages"] == len(seen) for result in again)


def test_saved_index_keeps_per_counts_and_rejects_another_field(tmp_path):
    path = tmp_path / "data.jselect"
    with Index.build(rows(), path=path, per="company_year") as index:
        assert (index.stats["per"], index.stats["per_values"]) == ("company_year", 4)
        assert index.stats["unique_passages"] == 71
        direct = index.select_per(task="complaints", scorer=Scorer(), encoding="bytes", tokens=700)
    saved = select_per(path, scorer=Scorer(), **OPTIONS)
    assert [r.context for r in saved] == [r.context for r in direct]
    # The same index still answers ordinary queries over the whole collection.
    assert select(path, task="complaints", scorer=Scorer(), encoding="bytes", tokens=700).items
    with pytest.raises(ValueError, match="not built with --per region"):
        select_per(path, task="complaints", per="region", scorer=Scorer())
    with Index.build(rows()) as plain:
        with pytest.raises(ValueError, match="not built with --per"):
            select_per(plain, scorer=Scorer(), **OPTIONS)
        with pytest.raises(ValueError, match="not built with --per"):
            plain.select_per(task="complaints")


def test_per_combines_with_group_by_and_rejects_bad_fields():
    turns = [
        {"thread": "t1", "company": "acme", "text": "strong: you charged me twice."},
        {"thread": "t2", "company": "bolt", "text": "strong: the app crashed."},
        {"thread": "t1", "company": "acme", "text": "strong: still no refund."},
    ]
    results = select_per(turns, task="complaints", per="company", group_by="thread", scorer=Scorer())
    assert [r.per["value"] for r in results] == ["acme", "bolt"]
    assert "charged me twice" in results[0].context and "still no refund" in results[0].context
    assert results[0].items[0].sources[0]["group_id"] == "t1"
    turns[2]["company"] = "bolt"
    with pytest.raises(ValueError, match="more than one --per value"):
        select_per(turns, task="complaints", per="company", group_by="thread", scorer=Scorer())
    for data in ([{"text": "no field"}], ["plain text"], [{"company": ["list"], "text": "bad type"}]):
        with pytest.raises(ValueError, match="--per"):
            select_per(data, task="complaints", per="company", scorer=Scorer())
    with pytest.raises(ValueError, match="per must name"):
        select_per(turns, task="complaints", per="", scorer=Scorer())


def test_async_entry_point_and_zero_budget():
    async def run():
        with pytest.raises(RuntimeError, match="aselect_per"):
            select_per(rows(), scorer=Scorer(), **OPTIONS)
        return await aselect_per(rows(), scorer=Scorer(), **OPTIONS)

    assert len(asyncio.run(run())) == 4
    empty = select_per(rows(), scorer=Scorer(), **{**OPTIONS, "tokens": 0})
    assert len(empty) == 4 and all(r.context == "" and r.stats["calls"] == 0 for r in empty)


def run(argv, fake=None):
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, transport=httpx.MockTransport(fake) if fake else None)
    return code, out.getvalue(), err.getvalue()


def test_cli_writes_one_json_object_per_group_and_scores_each_text_once(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    path = tmp_path / "complaints.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows()))
    calls = []

    async def fake(request):
        body = json.loads(request.content)
        calls.append(body)
        answers = {key: {"noul": 0.95 if "strong" in text else 0.05} for key, text in body["state"].items()}
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 10, "cost": 0.00001}})

    argv = ["complaints", str(path), "--per", "company_year", "--tokens", "300", "--seed", "2"]
    code, out, err = run([*argv, "--sample", "representative", "--json", "--stats"], fake)
    assert code == 0 and "4 contexts, one per company_year" in err
    sent = [text for call in calls for text in call["state"].values()]
    assert len(sent) == len(set(sent)) == 71
    lines = [json.loads(line) for line in out.splitlines()]
    assert [line["per"]["value"] for line in lines] == ["acme-2021", "bolt-2021", "cask-2021", "dune-2021"]
    single = json.loads(run(["complaints", str(path), "--json", "--tokens", "300"], fake)[1])
    for line in lines:
        assert set(line) == {*single, "per"} and line["schema_version"] == 1
        assert set(line["per"]) == {"field", "value", "passages", "occurrences"}
        assert line["tokens"] <= line["token_budget"] == 300
        assert line["stats"]["sample"] == "representative" and line["stats"]["scored_passages"] == 71
    code, out, err = run(argv, fake)
    assert code == 2 and "add --json" in err and not out
    target = tmp_path / "saved.jselect"
    code, out, _ = run(["index", str(path), "--per", "company_year", "-o", str(target), "--json"])
    assert code == 0 and json.loads(out)["stats"]["per_values"] == 4
    code, out, _ = run(
        ["complaints", str(target), "--per", "company_year", "--json", "--tokens", "300"], fake
    )
    assert code == 0 and len(out.splitlines()) == 4
    code, out, _ = run(["complaints", str(target), "--per", "region", "--json"], fake)
    assert code == 2 and "not built with --per region" in json.loads(out)["error"]["message"]
