"""Offline regressions for the remaining PR review findings."""

import io
import json
from pathlib import Path

import httpx
import pytest

from jselect import Index, Record, count_tokens, select, select_per
from jselect.cli import main
from jselect.judge import SemanticError
from jselect.select import render


def high(task, passages):
    return [0.9] * len(passages)


def sample(data, **options):
    options = {"encoding": "bytes", **options}
    return select(
        data,
        task="battery",
        sample="representative",
        scorer=high,
        **options,
    )


def fingerprint(result):
    # Source locations and rendered citations must track the input, not be byte-identical.
    keys = ("population_passages", "population_occurrences", "sample_occurrences", "sample_stop")
    return json.dumps(
        {
            "items": [(i.id, i.text, i.occurrences, i.draws) for i in result.items],
            "stats": {key: result.stats[key] for key in keys},
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode()


@pytest.mark.parametrize("change", ["filename", "absolute", "reverse", "repeat_positions"])
@pytest.mark.parametrize("long", [False, True])
@pytest.mark.parametrize("encoding", ["bytes", "o200k_base"])
def test_sample_is_bit_identical_after_location_changes(tmp_path, monkeypatch, change, long, encoding):
    rows = [f"battery report {i}: " + "detail " * (50 if long else 2) for i in range(110)]
    rows += [rows[0]] * 20
    first = tmp_path / "a.jsonl"
    second = tmp_path / (("long-name-" * 12 if change == "filename" else "b") + ".jsonl")
    first.write_text("\n".join(json.dumps({"text": text}) for text in rows))
    changed = list(reversed(rows)) if change == "reverse" else rows
    if change == "repeat_positions":
        changed = [rows[0]] * 21 + rows[1:110]
    second.write_text("\n".join(json.dumps({"text": text}) for text in changed))
    monkeypatch.chdir(tmp_path)
    target = first.name if change == "absolute" else second
    budget = 550 if encoding == "bytes" else 100
    for seed in range(8):
        before = sample(first, tokens=budget, seed=seed, encoding=encoding)
        after = sample(target, tokens=budget, seed=seed, encoding=encoding)
        assert before.items and after.items
        assert fingerprint(before) == fingerprint(after)
        for result in (before, after):
            assert result.tokens == count_tokens(result.context, encoding) <= budget
            for item in result.items:
                ref = item.sources[0]
                original = json.loads(Path(ref["source"]).read_text().splitlines()[ref["line"] - 1])["text"]
                assert item.text == original[ref["start"] : ref["end"]]


@pytest.mark.parametrize("limit", ["tokens", "max_items"])
def test_larger_sample_nests_when_fitted_population_is_unchanged(limit):
    rows = [Record(f"battery note {i}") for i in range(40)] + [Record("battery repeated")] * 40
    with Index.build(rows) as index:
        for seed in range(12):
            previous = None
            for size in [350, 500, 800, 1300, 2500] if limit == "tokens" else [1, 2, 4, 8, 16]:
                options = {"tokens": size} if limit == "tokens" else {"tokens": 100_000, "max_items": size}
                result = sample(index, seed=seed, **options)
                if previous:
                    assert [i.id for i in result.items[: len(previous.items)]] == [
                        i.id for i in previous.items
                    ]
                    assert all(
                        new.draws >= old.draws for old, new in zip(previous.items, result.items, strict=False)
                    )
                    assert result.stats["sample_occurrences"] >= previous.stats["sample_occurrences"]
                previous = result


@pytest.mark.parametrize("per", [False, True])
def test_sampling_fits_with_the_draw_header_reserved(per):
    text = "battery " + "x" * 120
    rows = [Record(text, id="r", source="toy", metadata={"g": "a"})] * 40
    # The passage fits alone without draws; fitting must now subdivide it to leave room for draws.
    with Index.build(rows, per="g" if per else None) as index:
        budget = count_tokens(render([next(index.all())]), "bytes")
        if per:
            result = select_per(
                index,
                per="g",
                task="battery",
                scorer=high,
                sample="representative",
                tokens=budget,
                encoding="bytes",
            )[0]
        else:
            result = sample(index, tokens=budget)
    assert result.items
    assert all(len(i.text) < len(text) for i in result.items)
    assert result.tokens <= budget
    assert not any("room reserved" in w for w in result.warnings)


def run(argv, fake=None):
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, transport=httpx.MockTransport(fake) if fake else None)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize(
    "options", [["--sample", "representative"], ["--per", "g"], ["--sample", "representative", "--per", "g"]]
)
def test_json_still_warns_about_local_population(tmp_path, options):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"g": "a", "text": "battery fault"}))
    code, out, err = run(["battery", str(path), "--json", "--encoding", "bytes", "--stats", *options])
    assert code == 0 and json.loads(out)["stats"]["mode"] == "local"
    assert "Local mode" in err
    if "--sample" in options and "--per" not in options:
        assert "keyword-matching occurrences" in err and "relevant occurrences" not in err
    code, out, _ = run(["battery", str(path), "--json", "--mode", "semantic", *options])
    assert code == 2 and "semantic mode needs" in json.loads(out)["error"]["message"]


@pytest.mark.parametrize("options", [{"sample": "representative"}, {"per": "g"}])
def test_full_scan_budget_advice_uses_supported_options(monkeypatch, options):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def forbidden(request):
        raise AssertionError("preflight must not send a request")

    with pytest.raises(SemanticError) as caught:
        (select_per if "per" in options else select)(
            [{"g": "a", "text": "battery fault"}],
            task="battery",
            mode="semantic",
            encoding="bytes",
            budget=0.00000001,
            cache=False,
            _transport=httpx.MockTransport(forbidden),
            **options,
        )
    message = str(caught.value)
    assert "--candidates" not in message and "shortlist" not in message
    assert "raise --budget" in message and "smaller collection" in message


@pytest.mark.parametrize("sample_mode", [False, True])
def test_zero_token_per_keeps_each_value_without_scoring(tmp_path, sample_mode):
    rows = [{"g": "z", "text": "battery"}, {"g": "blank", "text": " "}, {"g": "a", "text": "battery"}]

    def forbidden(*args):
        raise AssertionError("zero-token contexts must not be scored")

    options = {"sample": "representative"} if sample_mode else {}
    results = select_per(
        rows, task="battery", per="g", tokens=0, encoding="bytes", scorer=forbidden, **options
    )
    assert [r.per["value"] for r in results] == ["z", "blank", "a"]
    assert all(r.items == [] and r.context == "" and r.tokens == r.stats["calls"] == 0 for r in results)
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    code, out, _ = run(
        [
            "battery",
            str(path),
            "--tokens",
            "0",
            "--per",
            "g",
            "--json",
            "--encoding",
            "bytes",
            *(["--sample", "representative"] if sample_mode else []),
        ]
    )
    assert code == 0 and [json.loads(line)["per"]["value"] for line in out.splitlines()] == [
        "z",
        "blank",
        "a",
    ]


@pytest.mark.parametrize("seed", ["0", "7"])
def test_cli_seed_requires_sampling(tmp_path, seed):
    path = tmp_path / "rows.txt"
    path.write_text("battery fault")
    code, out, _ = run(["battery", str(path), "--local", "--json", "--seed", seed])
    assert code == 2 and "--seed requires --sample" in json.loads(out)["error"]["message"]


@pytest.mark.parametrize("options", [["--sample", "representative"], ["--per", "g"]])
def test_expired_key_fails_instead_of_falling_back(tmp_path, monkeypatch, options):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"g": "a", "text": "battery fault"}))
    code, out, _ = run(
        ["battery", str(path), "--json", "--encoding", "bytes", *options],
        lambda request: httpx.Response(401, json={"error": "expired"}),
    )
    assert code == 2 and "HTTP 401" in json.loads(out)["error"]["message"]


@pytest.mark.parametrize("encoding", ["bytes", "o200k_base"])
def test_sample_citations_use_content_ids_when_locations_exceed_the_allowance(encoding):
    rows = [Record("battery fault", source="source-東京-" * 100, id="record-" * 100)]
    result = select(rows, task="battery", scorer=high, sample="representative", encoding=encoding, tokens=300)
    assert result.items and result.tokens == count_tokens(result.context, encoding) <= 300
    assert result.context.startswith('[1] {"passage":"' + result.items[0].id + '"}')
    assert result.items[0].sources[0]["source"] == rows[0].source
    short = sample([Record("battery fault", source="s", id="r")], tokens=300)
    assert short.context.startswith('[1] {"source":"s"')
