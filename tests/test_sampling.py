import io
import json

import httpx
import pytest
from golden import GOLDEN, default_outputs

from jselect import Index, Passage, Record, count_tokens, select
from jselect.cli import main
from jselect.judge import SemanticError
from jselect.select import Draw, render


def graded(task, passages):
    """Custom scorer: 'strong' passages are clearly relevant, 'weak' ones fall between the two defaults."""
    return [0.9 if "strong" in p.text else 0.4 if "weak" in p.text else 0.05 for p in passages]


def complaints(strong=40, weak=10, noise=10):
    rows = [Record(f"strong complaint {i}: the refund never arrived.", id=f"s{i}") for i in range(strong)]
    rows += [Record(f"weak complaint {i}: the refund page is slow.", id=f"w{i}") for i in range(weak)]
    return rows + [Record(f"unrelated note {i} about the office party.", id=f"n{i}") for i in range(noise)]


def sample(data, **options):
    options = {"task": "refund complaints", "encoding": "bytes", "scorer": graded, **options}
    return select(data, sample="representative", **options)


def test_default_selection_is_byte_identical_to_the_previous_release():
    assert default_outputs() == GOLDEN.read_text(encoding="utf-8")


def test_same_seed_repeats_the_draw_and_other_seeds_differ():
    with Index.build(complaints()) as index:
        runs = {seed: sample(index, tokens=600, seed=seed) for seed in (0, 1, 2, 3)}
        again = sample(index, tokens=600, seed=2)
    assert again.context == runs[2].context
    assert [i.id for i in again.items] == [i.id for i in runs[2].items]
    assert len({r.context for r in runs.values()}) == len(runs)
    # The draw does not depend on the order in which records were indexed or scanned.
    assert sample(complaints()[::-1], tokens=600, seed=2).context == runs[2].context


def test_items_and_stats_describe_the_sampling_rule():
    result = sample(complaints(), tokens=600, seed=5)
    assert result.items and result.stats["selected"] == len(result.items)
    for item in result.items:
        assert item.novelty is None and item.draws == 1
        assert "random draw" in item.reason and "0.5" in item.reason and "custom" in item.reason
    stats = result.stats
    assert stats["selection_method"] == "random_occurrence_sample_without_replacement"
    assert (stats["sample"], stats["seed"], stats["threshold"], stats["scan"]) == (
        "representative",
        5,
        0.5,
        "all",
    )
    assert "diversity" not in stats and "candidate_limit" not in stats
    payload = result.to_dict()
    assert payload["schema_version"] == 1
    assert payload["items"][0]["draws"] == 1 and payload["items"][0]["novelty"] is None


def test_threshold_defines_the_population():
    data = complaints(strong=40, weak=10, noise=10)
    default = sample(data, tokens=100_000)
    assert default.stats["population_passages"] == default.stats["population_occurrences"] == 40
    assert default.stats["sample_stop"] == "population" and len(default.items) == 40
    assert all("strong" in item.text for item in default.items)
    lower = sample(data, tokens=100_000, threshold=0.3)
    assert lower.stats["population_passages"] == 50 and lower.stats["threshold"] == 0.3
    assert sum("weak" in item.text for item in lower.items) == 10
    nothing = sample(data, tokens=100_000, threshold=0.95)
    assert nothing.items == [] and nothing.stats["population_passages"] == 0
    assert nothing.stats["sample_occurrences"] == 0
    assert any("No new passage" in warning for warning in nothing.warnings)


@pytest.mark.parametrize("encoding", ["o200k_base", "bytes"])
@pytest.mark.parametrize("budget", [1, 60, 150, 400, 1600])
def test_budget_is_never_exceeded_and_counts_are_reported(encoding, budget):
    rows = [Record(f"strong café 東京 👩🏽‍💻 report {i}. " * (1 + i % 9), id=str(i)) for i in range(60)]
    rows += [Record("strong repeated complaint about a locked account.", id=f"r{i}") for i in range(25)]
    for seed in range(8):
        result = sample(rows, tokens=budget, encoding=encoding, seed=seed, chunk_size=200, overlap=30)
        assert result.tokens == count_tokens(result.context, encoding) <= budget
        stats = result.stats
        assert stats["sample_passages"] == len(result.items) <= stats["population_passages"]
        assert stats["sample_occurrences"] == sum(i.draws for i in result.items)
        assert stats["sample_occurrences"] <= stats["population_occurrences"]
        assert all(1 <= item.draws <= item.occurrences for item in result.items)
        assert stats["sample_stop"] in {"budget", "population"}


def test_population_counts_passages_and_occurrences():
    rows = [Record("strong repeated complaint about a locked account.", id=f"r{i}") for i in range(25)]
    rows += complaints(strong=30, weak=5, noise=5)
    census = sample(rows, tokens=100_000)
    assert census.stats["population_passages"] == 31
    assert census.stats["population_occurrences"] == 55
    # A census draws every occurrence, and the context shows the repeated text once with its count.
    assert census.stats["sample_occurrences"] == 55 and census.stats["sample_stop"] == "population"
    repeated = next(item for item in census.items if item.occurrences == 25)
    assert repeated.draws == 25 and census.context.count("locked account") == 1
    assert '"draws":25}' in census.context
    partial = sample(rows, tokens=900, seed=4)
    assert partial.stats["population_occurrences"] == 55 and partial.stats["sample_stop"] == "budget"
    assert 0 < partial.stats["sample_occurrences"] < 55


def test_a_passage_seen_fifty_times_is_drawn_about_fifty_times_as_often():
    rows = [Record("strong duplicate: the account was locked.", id=f"d{i}") for i in range(50)]
    rows += [Record(f"strong single {i:03d}: the account was locked.", id=f"u{i}") for i in range(150)]
    repeated = singles = 0
    shares = []
    with Index.build(rows) as index:
        for seed in range(500):
            result = sample(index, tokens=1000, seed=seed)
            drawn = sum(item.draws for item in result.items if item.occurrences == 50)
            repeated += drawn
            singles += result.stats["sample_occurrences"] - drawn
            shares.append(drawn / result.stats["sample_occurrences"])
    # Seeds are fixed, so this is deterministic. 50 is the design value; repeats of an included
    # text add no tokens, which lifts the measured ratio slightly (see the README's stated limits).
    assert 42 < repeated / (singles / 150) < 62
    # The repeated text holds a quarter of all relevant occurrences, and of the average sample.
    assert 0.22 < sum(shares) / len(shares) < 0.28


def test_draw_stops_at_the_first_passage_that_does_not_fit():
    rows = [Record(f"strong short note {i}.", id=f"s{i}") for i in range(30)]
    rows += [Record(f"strong long report {i}. " + "detail " * 110, id=f"l{i}") for i in range(10)]
    stopped_with_room = 0
    with Index.build(rows) as index:
        for seed in range(40):
            census = sample(index, tokens=1_000_000, seed=seed).items
            result = sample(index, tokens=1500, seed=seed)
            taken = [item.text for item in result.items]
            # The sample is the front of the seeded order: nothing is skipped to make room.
            assert taken == [item.text for item in census[: len(taken)]]
            assert result.stats["sample_stop"] == "budget"
            assert count_tokens(render([*result.items, census[len(taken)]]), "bytes") > 1500
            smallest = min(count_tokens(render([i]), "bytes") for i in census if "short" in i.text)
            stopped_with_room += 1500 - result.tokens >= smallest + 2
    # Some draws end with room a short passage could have filled; filling it would favor short text.
    assert stopped_with_room > 0


def test_max_items_fixes_the_sample_size():
    result = sample(complaints(), tokens=100_000, max_items=7, seed=1)
    assert len(result.items) == 7 and result.stats["sample_stop"] == "max_items"
    assert result.stats["sample_passages"] == 7 and result.stats["population_passages"] == 40


def test_against_is_excluded_from_the_population_and_continues_the_same_order():
    with Index.build(complaints()) as index:
        order = [item.text for item in sample(index, tokens=1_000_000, seed=9).items]
        first = sample(index, tokens=500, seed=9)
        second = sample(index, tokens=500, seed=9, against=first)
    assert first.items and second.items
    assert not {i.text for i in first.items} & {i.text for i in second.items}
    assert second.stats["population_passages"] == 40 - len(first.items)
    assert second.stats["previous_passages"] == len(first.items)
    taken = [item.text for item in [*first.items, *second.items]]
    assert taken == order[: len(taken)]


def test_local_population_is_every_lexical_match_not_the_shortlist():
    rows = [f"Battery fault observed on device {i}." for i in range(400)] + ["Garden flowers bloom."] * 3
    with Index.build(rows) as index:
        result = select(
            index, task="battery fault", mode="local", sample="representative", tokens=800, candidates=8
        )
        default = select(index, task="battery fault", mode="local", tokens=800, candidates=8)
    assert result.stats["population_passages"] == result.stats["passages_evaluated"] == 400
    assert (result.stats["mode"], result.stats["scan"], result.stats["threshold"]) == ("local", "all", 0.0)
    assert result.stats["source_passages_considered"] == 401
    assert any("share a task term" in warning for warning in result.warnings)
    assert not any("shortlisted" in warning for warning in result.warnings)
    assert default.stats["scan"] == "shortlist" and default.stats["passages_evaluated"] == 8
    # A sample of the matches reaches past the head of the lexical ranking.
    assert max(int(item.text.split()[-1].rstrip(".")) for item in result.items) > 8


def test_full_scan_streams_and_keeps_only_the_front_of_the_order():
    calls = []

    def scorer(task, passages):
        calls.append(len(passages))
        return [0.9] * len(passages)

    result = sample((f"strong record {i}" for i in range(3000)), tokens=400, scorer=scorer)
    assert max(calls) <= 256 and sum(calls) == 3000
    assert result.stats["passages_evaluated"] == result.stats["population_passages"] == 3000
    draw = Draw(tokens=400, encoding="bytes", seed=0)
    with Index.build(f"strong record {i}" for i in range(3000)) as index:
        for passage in index.all():
            draw.add(passage)
        assert len(draw.head) < 40 and draw.passages == 3000


def test_equal_excerpts_reaching_the_draw_separately_become_one_item():
    def passage(record, parent, occurrences):
        ref = {"source": "a.txt", "line": 1, "record_id": record, "start": 0, "end": 17, "passage_id": parent}
        return Passage("same-id", "strong shared text", [ref], occurrences, 0.9)

    for seed in range(20):
        # Cut from two indexed passages: two independent units, one item once both are drawn.
        draw = Draw(tokens=10_000, encoding="bytes", seed=seed)
        draw.extend([passage("first", "parent-1", 3), passage("second", "parent-2", 4)])
        items, stats = draw.finish("rule")
        assert len(items) == 1 and items[0].draws == items[0].occurrences == 7
        assert sorted(ref["record_id"] for ref in items[0].sources) == ["first", "second"]
        assert (stats["population_passages"], stats["sample_occurrences"]) == (2, 7)
        # The same unit offered twice is one unit with the combined count.
        again = Draw(tokens=10_000, encoding="bytes", seed=seed)
        again.extend([passage("first", "parent-1", 3), passage("first", "parent-1", 4)])
        items, stats = again.finish("rule")
        assert (stats["population_passages"], stats["population_occurrences"], items[0].draws) == (1, 7, 7)


@pytest.mark.parametrize(
    "options",
    [{"sample": "random"}, {"sample": "representative", "scan": "shortlist"}]
    + [{"sample": "representative", "seed": seed} for seed in (1.5, True, "7")],
)
def test_invalid_sampling_options(options):
    with pytest.raises(ValueError):
        select(["text"], task="test", scorer=graded, **options)


class Fake:
    def __init__(self):
        self.calls = []

    async def __call__(self, request):
        body = json.loads(request.content)
        self.calls.append(body)
        answers = {
            key: {"noul": 0.95 if "relevant" in text else 0.4 if "maybe" in text else 0.05}
            for key, text in body["state"].items()
        }
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 10, "cost": 0.00001}})


def run(argv, fake=None):
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, transport=httpx.MockTransport(fake) if fake else None)
    return code, out.getvalue(), err.getvalue()


def test_cli_semantic_sample_reports_methods_and_keeps_the_budget_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    path = tmp_path / "tickets.jsonl"
    texts = [f"relevant ticket {i}" for i in range(30)] + [f"maybe ticket {i}" for i in range(10)]
    texts += [f"noise {i}" for i in range(10)]
    path.write_text("\n".join(json.dumps({"text": text}) for text in texts))
    fake = Fake()
    argv = ["trust", str(path), "--sample", "representative", "--seed", "11", "--tokens", "300"]
    code, out, err = run([*argv, "--json", "--stats"], fake)
    assert code == 0
    result = json.loads(out)
    assert len({text for call in fake.calls for text in call["state"].values()}) == 50
    stats = result["stats"]
    assert (stats["population_passages"], stats["threshold"], stats["seed"]) == (30, 0.5, 11)
    assert stats["passages_evaluated"] == 50 and stats["scan"] == "all"
    assert result["tokens"] <= 300 and all("relevant" in item["text"] for item in result["items"])
    line = next(line for line in err.splitlines() if "representative sample" in line)
    assert f"{stats['sample_occurrences']} of 30 relevant occurrences" in line
    assert "threshold 0.5; seed 11; random without replacement; ended by budget" in line
    # The default rule on the same scores admits the 0.4 passages; the sample's population does not.
    code, out, _ = run(["trust", str(path), "--json", "--tokens", "100000", "--candidates", "64"], fake)
    assert sum("maybe" in item["text"] for item in json.loads(out)["items"]) == 10
    code, plain, _ = run(argv, fake)
    assert code == 0 and plain == result["context"]
    code, out, _ = run([*argv, "--scan", "shortlist", "--json"], fake)
    assert code == 2 and "omit --scan shortlist" in json.loads(out)["error"]["message"]

    def forbidden(request):
        raise AssertionError("network request before complete budget check")

    with pytest.raises(SemanticError, match="budget"):
        select(
            (f"long record {i} " * 40 for i in range(1000)),
            task="records",
            mode="semantic",
            sample="representative",
            budget=0.0001,
            cache=False,
            _transport=httpx.MockTransport(forbidden),
        )
