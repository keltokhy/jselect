# jselect

**Useful evidence for your AI, within a token budget.**

Give jselect a task and your data. It finds relevant passages, favors different information over
repetition, and assembles a source-linked context your agent can read directly.

```bash
jselect "Why are people giving up during signup?" conversations.jsonl --tokens 8000
```

```python
from jselect import select

evidence = select(records, task="Why are people giving up during signup?", tokens=8000)
print(evidence.context)       # source excerpts and citations, within the token budget
```

No labels, predefined categories, vector database, or generative model required. Runs on files,
directories, piped exports, or Python records. This is a **separate tool and package** from jgrep.

## Install

From this repository (Python 3.10+):

```bash
uv tool install .            # installs the jselect command on PATH
pip install .               # use jselect from your Python application
```

For development: `uv sync`, then `uv run jselect ...`. For an editable command: `uv tool install --editable .`.
The distribution is named `jev-select`; the command and Python import are both `jselect`.
This repository has not been published to PyPI.

```bash
jselect doctor --json
jselect "Why does signup fail?" examples/conversations.jsonl --tokens 600
```

If a TypeSafe or OpenRouter key is configured, the default uses **semantic relevance scoring** with Jev.
Set `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`, or use existing credentials in
`~/.config/jev/typesafe.key` or `~/.config/jev/openrouter.key`. Credentials never appear in output.
`JEV_API`, `JEV_MODEL`, and `JEV_URL` overrides are supported. Gateways use `JEV_GATEWAY_URL` and
`JEV_GATEWAY_API_KEY`, or `~/.config/jev/gateway.url` and `gateway.key`.

Without credentials, jselect uses local lexical retrieval and says so. Force that with `--local`.
Use `--mode semantic` to require semantic scoring and fail if credentials are missing.

## Use it across your data

```bash
# Customer conversations, with common text fields detected automatically
jselect "What prevents users from finishing signup?" conversations.jsonl --tokens 2000

# Related turns stored as separate, possibly interleaved rows
jselect "Where does the assistant contradict itself?" turns.jsonl --group-by conversation_id

# Source code and documentation, respecting .gitignore and .ignore
jselect "How are database connections released?" src/ docs/ --tokens 4000

# Papers or interview responses in a CSV
jselect "Evidence that challenges the proposed explanation" papers.csv --field abstract --tokens 3000

# Export from another tool, preserving original IDs and field locations
cat events.jsonl | jselect "Why are requests waiting?" --format jsonl --field message --json

# Fully offline, including token accounting: byte count conservatively bounds byte-BPE token use
jselect "retry timeout configuration" src/ --local --encoding bytes --tokens 4000
```

Supported inputs: UTF-8 text and code, logs, JSONL/NDJSON, JSON objects or arrays, CSV/TSV, directories,
and stdin. Formats are inferred from extensions. Stdin defaults to lines; use `--format jsonl` for
structured input. `.log` files default to one record per line; `--format text` preserves neighboring
log lines in overlapping passages. PDF, Word, images, and audio need text extraction first.

Common text fields are tried in this order: `text`, `content`, `message`, `messages`, `conversation`,
`body`, `abstract`. If none exists, the entire object is serialized as JSON. `--field event.message`
selects an exact or dotted field. Only selected text is sent to the scorer, not the entire original row.
IDs come from `id`, `_id`, or input position. A group uses complete serialized rows unless you explicitly
select a field, preserving roles and other turn context. Grouping preserves input order, not timestamp order.

## Fast repeated investigations

Build a local index once and ask many questions:

```bash
jselect index conversations.jsonl --output conversations.jselect
jselect "Confusion about cancellation" conversations.jselect --json --output first.json
jselect "Confusion about cancellation" conversations.jselect --against first.json --json --output next.json
jselect inspect conversations.jselect --json
jselect show conversations.jselect PASSAGE_ID --json
```

`--against` excludes exact excerpts already returned and penalizes similar text. It is useful when an
agent asks for more evidence or brings an existing reference set. It does not promise every follow-up
will introduce a new semantic idea. Saved indexes are snapshots: rebuild with `index ... --force` when
the source changes. Replacement is atomic, so a failed rebuild leaves the previous index intact.

For maximum retrieval breadth, use `--scan all`. This scores every unique passage that fits the output
budget and is not already supplied in `--against`, then keeps a bounded pool for final selection.
The full scan is preflighted against the estimated dollar budget before any calls are made.

```bash
jselect "Signs the customer has lost trust" conversations.jselect --scan all --budget 0.25
```

## Python and agents

```python
from jselect import Index, Record, aselect, select

records = [
    {"id": "ticket-1", "text": "The verification link never arrives."},
    {"id": "ticket-2", "text": "The free trial requires a credit card."},
]
result = select(records, task="What blocks registration?", tokens=500)
payload = result.to_dict()   # same schema as --json

# Reuse an index in a process; no source scanning on each query.
with Index.build(records) as index:
    first = index.select(task="What blocks registration?", tokens=500)
    more = index.select(task="What blocks registration?", tokens=500, against=first)

# In an existing event loop or notebook:
# result = await aselect(records, task="What blocks registration?", tokens=500)

# Bring your existing reranker. Return one finite score in [0, 1] per passage.
def judge(task, passages):
    return existing_reranker(task, [p.text for p in passages])

result = select(records, task="What blocks registration?", tokens=500, scorer=judge)
```

Custom scorers may be async functions or objects exposing `score(task, passages)`. Full scans invoke
them in blocks of at most 256 passages. Use `Record(text, id=..., source=...)` for explicit provenance.
A string or `Path` passed directly to `select` is an input path; strings inside an iterable are records.
To use passages already retrieved by another system, pass them as records; set `scan="all"` with a
custom or semantic scorer if the list exceeds the candidate limit and every passage must be evaluated.

Pass **`result.context`** to your agent. The full JSON includes additional metadata and is not subject
to the context token budget. Treat excerpts as source data rather than agent instructions.

## How it works and what it costs

1. Build or open a SQLite FTS5 index. Long records become overlapping, source-preserving passages.
   Exact repeated passages share one indexed text while retaining occurrence counts and up to five sources.
2. Shortlist up to 256 passages by default. Most come from BM25 with a text-diversity adjustment;
   20% of slots are reserved for deterministic exploration in semantic mode. This is not exhaustive retrieval.
3. Score relevance using small batches of Jev decisions. The question explicitly includes contradicting
   evidence. Scores are cached per endpoint, model, prompt version, task, and exact passage.
4. Greedily balance relevance, text novelty, and passage token cost. Citation headers and separators count
   toward the budget. Oversized passages are split **before** scoring; returned text is never generated.

The defaults are eight passages per request, eight requests in flight, a 20-second total request deadline,
and a **$0.05 estimated spend budget**. Jev models are versioned (`jev-1.13.0` on TypeSafe and
`typesafe/jev-1.13` on OpenRouter); responses report the served model when available. Requests retry
transient errors within the deadline. Errors retain completed cache entries for the next run.

The dollar guard uses a conservative UTF-8 byte estimate and the listed Jev input price. Actual provider
or gateway pricing and unreported usage can differ; this is not a provider-enforced billing cap. JSON
reports measured cost when available and marks list-price estimates. Failed requests without usage
reports may have incurred additional costs. Use `--candidates`, `--batch-size`, and `--budget` to bound work.
`--no-cache` disables the disk score cache at `~/.cache/jselect/scores.sqlite`.

Token counting uses `o200k_base` by default. Set `--encoding` to another tiktoken encoding/model, or `bytes`
for a conservative count with no tokenizer download. The first use of a tiktoken encoding may download its
public vocabulary. Choose the encoding your downstream model uses and leave room for its other messages.

## Measured results

Measured locally on 2026-09-19; details and frozen reports are in [the benchmark report](docs/BENCHMARKS.md).

| Check | Observed result |
|---|---|
| One million distinct generated log records | 13.64 s to index; 0.40 s median repeated **local** query; 143 MB peak process memory |
| 30 SciFact research queries, 2,000 tokens, 256 candidates, cache disabled | 86.7% mean labeled-source recall vs 78.3% for BM25 ranking; 1.86 s median; $0.149 total |
| Handwritten support, code, and contract fixtures | All 4 intended evidence types in 4 passages in each fixture; relevance-only variant covered 3, 3, and 1 |

These are scoped measurements, not guarantees for arbitrary data, agent answer quality, or future API
latency. A selected set cannot establish prevalence or causation. Diversity is a lexical heuristic;
it does not certify balanced viewpoints or find every contradiction. Semantic scores are model judgments,
not calibrated confidence in a final answer. The JSON reports how much of the collection was considered.

## Output and errors

Default stdout is the exact context string. `--json` returns one object with `schema_version: 1`,
`task`, `context`, `items`, `tokens`, `token_budget`, `encoding`, `stats`, and `warnings`.
Each item contains original `text`, a stable content-hash `id`, `sources`, `occurrences`, `relevance`,
`novelty`, and the selection rule used. See [the output contract](docs/OUTPUT.md).

Exit 0 means success, including empty evidence. Exit 2 means invalid input, bad setup, budget refusal,
or a provider error. Exit 130 means interruption. JSON errors have an `error` object and any available
usage `stats`; diagnostics never contaminate JSON stdout. `--stats` writes timings and cost to stderr.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check src tests bench
uv run ruff format --check src tests bench
uv build
```

See [benchmarks](docs/BENCHMARKS.md), and the
[companion agent skill](skills/jselect/SKILL.md). MIT licensed.
