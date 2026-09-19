import asyncio
import io
import json

import httpx
import pytest

from jselect import Passage, select
from jselect.cli import main
from jselect.judge import Backend, JevScorer, SemanticError


class Fake:
    def __init__(self, status=200):
        self.calls = []
        self.status = status

    async def __call__(self, request):
        body = json.loads(request.content)
        self.calls.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "SECRET must never be printed"})
        await asyncio.sleep(0.001)
        answers = {key: {"noul": 0.95 if "relevant" in text else 0.05} for key, text in body["state"].items()}
        return httpx.Response(
            200,
            json={
                "answers": answers,
                "model": "fixture-pinned",
                "usage": {"input_tokens": 100, "cost": 0.00001},
            },
        )


def run(argv, fake=None):
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, transport=httpx.MockTransport(fake) if fake else None)
    return code, out.getvalue(), err.getvalue()


def test_semantic_scoring_batches_and_cached_results(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake = Fake()
    values = [f"relevant issue {i}" for i in range(19)] + ["noise unrelated"]
    first = select(values, task="issues", mode="semantic", batch_size=8, _transport=httpx.MockTransport(fake))
    assert len(fake.calls) == 3
    assert first.stats["scored_passages"] == 20
    assert not any(p.text == "noise unrelated" for p in first.items)
    second = select(
        values, task="issues", mode="semantic", batch_size=3, _transport=httpx.MockTransport(fake)
    )
    assert len(fake.calls) == 3
    assert second.stats["calls"] == 0 and second.stats["cost"] == 0
    assert second.stats["cached_passages"] == 20


def test_low_budget_fails_before_network(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake = Fake()
    with pytest.raises(SemanticError, match="budget"):
        select(
            ["relevant" * 100],
            task="test",
            mode="semantic",
            budget=0.00000001,
            _transport=httpx.MockTransport(fake),
        )
    assert not fake.calls


def test_cache_isolated_by_model_and_endpoint(tmp_path):
    fake = Fake()
    path = tmp_path / "cache.sqlite"
    passage = Passage("one", "relevant", [])

    async def exercise(url, model):
        scorer = JevScorer(
            Backend("fixture", url, model, "key", "env"), cache_path=path, transport=httpx.MockTransport(fake)
        )
        try:
            return await scorer.score("task", [passage])
        finally:
            scorer.close()

    for url, model in [("https://a.test", "v1"), ("https://a.test", "v2"), ("https://b.test", "v2")]:
        assert asyncio.run(exercise(url, model)) == [0.95]
    assert len(fake.calls) == 3


@pytest.mark.parametrize("bad", [None, [], {"p0": {"noul": 0.9}, "p1": {"noul": "bad"}}])
def test_invalid_batch_is_never_partially_cached(tmp_path, bad):
    scorer = JevScorer(
        Backend("fixture", "https://test", "v1", "key", "env"),
        cache_path=tmp_path / "cache.sqlite",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": bad})),
    )
    try:
        with pytest.raises(SemanticError):
            asyncio.run(scorer.score("task", [Passage("a", "text", []), Passage("b", "other", [])]))
        assert scorer.db.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0
    finally:
        scorer.close()


def test_cli_doctor_no_key_and_json_errors(tmp_path):
    code, out, err = run(["--json", "doctor"])
    assert code == 0 and not err
    assert json.loads(out)["local_ready"] and not json.loads(out)["semantic_configured"]
    code, out, err = run(["anything", "--tokens", "nonsense", "--json"])
    assert code == 2 and "error" in json.loads(out) and not err
    code, out, err = run(["test", str(tmp_path / "missing"), "--json"])
    assert code == 2 and "does not exist" in json.loads(out)["error"]["message"]


def test_cli_semantic_provider_errors_do_not_leak_response(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    path = tmp_path / "data.txt"
    path.write_text("relevant source")
    code, out, err = run(["test", str(path), "--json"], Fake(status=401))
    assert code == 2 and "HTTP 401" in json.loads(out)["error"]["message"]
    assert "SECRET" not in out + err and "test-key" not in out + err


def test_cli_saved_index_and_show(tmp_path):
    source, target = tmp_path / "input.txt", tmp_path / "data.jselect"
    source.write_text("How to resolve a database deadlock.")
    code, out, err = run(["index", str(source), "--output", str(target), "--json"])
    assert code == 0 and json.loads(out)["stats"]["records"] == 1
    code, out, err = run(["database deadlock", str(target), "--local", "--json"])
    assert code == 0
    result = json.loads(out)
    code, out, err = run(["show", str(target), result["items"][0]["id"], "--json"])
    assert code == 0 and json.loads(out)["passage"]["text"] == source.read_text()


def test_local_mode_does_not_use_configured_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    path = tmp_path / "data.txt"
    path.write_text("database deadlock")
    fake = Fake()
    code, out, err = run(["database", str(path), "--local", "--json"], fake)
    assert code == 0 and json.loads(out)["stats"]["mode"] == "local" and not fake.calls


def test_flags_can_precede_or_follow_inputs_and_unknown_tokenizer_is_json(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("refund policy")
    code, out, err = run(["refund", "--tokens", "200", str(path), "--local", "--json"])
    assert code == 0 and json.loads(out)["items"]
    code, out, err = run(["refund", str(path), "--encoding", "not-a-tokenizer", "--json"])
    assert code == 2 and "unknown tokenizer" in json.loads(out)["error"]["message"]


def test_closed_pipeline_is_success(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("refund policy")

    class ClosedPipe:
        def write(self, value):
            raise BrokenPipeError()

    err = io.StringIO()
    assert main(["refund", str(path), "--local"], out=ClosedPipe(), err=err) == 0
    assert not err.getvalue()


def test_invalid_gateway_setup_is_actionable_and_does_not_echo_credentials(monkeypatch):
    monkeypatch.setenv("JEV_GATEWAY_API_KEY", "fixture-secret")
    monkeypatch.setenv("JEV_GATEWAY_URL", "not-a-url")
    code, out, err = run(["doctor", "--api", "gateway", "--json"])
    result = json.loads(out)
    assert code == 0 and not result["semantic_configured"]
    assert "complete HTTP or HTTPS URL" in result["hint"]
    assert "fixture-secret" not in out + err
