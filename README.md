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

Python 3.10+:

```bash
uv tool install jev-select   # installs the jselect command on PATH
pip install jev-select       # use jselect from your Python application
```

The distribution is named `jev-select`; the command and Python import are both `jselect`.

```bash
jselect doctor --json
echo 'The signup verification email never arrived.' | jselect "signup problems" --tokens 600
```

For development, clone [the repository](https://github.com/keltokhy/jselect), run `uv sync`,
then `uv run jselect ...`. For an editable command: `uv tool install --editable .`.

If a TypeSafe or OpenRouter key is configured, the default uses **semantic relevance scoring** with Jev.
Set `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`, or use existing credentials in
`~/.config/jev/typesafe.key` or `~/.config/jev/openrouter.key`. Credentials never appear in output.
`JEV_API`, `JEV_MODEL`, and `JEV_URL` overrides are supported. Gateways use `JEV_GATEWAY_URL` and
`JEV_GATEWAY_API_KEY`, or `~/.config/jev/gateway.url` and `gateway.key`.

Without credentials, jselect uses local lexical retrieval with a shortlist and says so. Force that with `--local`.
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

Semantic and custom scoring default to `--scan all`. This scores every eligible passage across the
collection, splitting oversized passages to fit the output budget and excluding excerpts already supplied
in `--against`, then keeps a bounded pool for final selection. The full semantic scan is preflighted against
the estimated dollar budget before any calls are made. If it exceeds the budget, the command fails without
making scoring requests; it never silently switches to a shortlist.

```bash
jselect "Signs the customer has lost trust" conversations.jselect --budget 0.25

# Explicitly trade retrieval coverage for less scoring work
jselect "Signs the customer has lost trust" conversations.jselect --scan shortlist --candidates 256
```

`--candidates` defaults to 256 and limits the pool retained **after** scoring in a full scan. With
`--scan shortlist`, it limits passages evaluated **before** final selection. Local mode defaults to a
lexical shortlist; explicitly requesting `--scan all` requires a semantic or custom scorer.

## Representative samples

The default rule returns the most relevant and most different passages. That serves an agent looking
for leads, and it is the wrong input for measurement: a context built from each company's most striking
complaints shows how bad the worst of them are, not what is typical. `--sample representative` replaces
the selection rule with a random draw:

```bash
jselect "Complaints about account access" complaints.jsonl --sample representative --seed 7 --tokens 4000 --stats
```

1. The population is every passage judged relevant: relevance at or above `--threshold`, which defaults
   to 0.5 in this mode. The scan is always complete, so `--scan shortlist` is refused; the dollar guard
   and preflight work as usual. Excerpts supplied with `--against` leave the population.
2. Passage occurrences are drawn uniformly at random, without replacement, in an order fixed by `--seed`
   (default 0). An exact repeated passage is indexed once, so a passage seen 50 times has 50 chances.
   It appears once in the context with the number of its occurrences that were drawn: `"draws":3` in
   the citation header and `draws` in JSON.
3. The draw ends at the first passage that does not fit the remaining token budget, at `-n` passages,
   or when the population runs out. Nothing is skipped to make room, because filling the gap with
   whatever fits would favor short passages. Items keep the order of the draw, which carries no ranking.
   Novelty, `--diversity`, and `--candidates` play no part.

`--stats` adds a line for a methods section, and the same numbers are in the JSON `stats`:

```
jselect: representative sample: N_DRAWN of N_RELEVANT relevant occurrences (K_DRAWN of K_RELEVANT passages); threshold 0.5; seed 7; random without replacement; ended by budget
```

```python
result = select(records, task="Complaints about account access", sample="representative", seed=7)
drawn, relevant = result.stats["sample_occurrences"], result.stats["population_occurrences"]
```

What the sample does and does not support:

- The unit is a passage occurrence, not a record, a customer, or an event. A long record yields several
  overlapping passages and so has more chances than a short one.
- The passage that ends the draw is left out, and it is more often a long one. Long passages are
  therefore somewhat under-represented, more so when one passage takes a large share of the budget.
  To avoid this, fix the sample size with `-n` and give `--tokens` enough room that
  `stats.sample_stop` is `max_items` rather than `budget`.
- Further draws of a text already in the context add almost no tokens and never end the draw, so a
  heavily repeated text is drawn slightly more often than its share of occurrences.
- "Relevant" is the scorer's judgment at the threshold; scoring errors move the population. With
  `--local` no model judges relevance: the population is every passage sharing at least one task term
  after stop-word removal and stemming, and `--threshold` (default 0) applies to the normalized
  lexical score. Report that as a keyword match.
- The same index, task, budget, encoding, threshold, and seed give the same sample. With `--against`
  and the same seed, the draw continues down the same order past the excerpts already returned.
- A small sample is noisy, and a representative sample still cannot establish causation. `stats`
  reports both sizes so that uncertainty can be stated.

## Python and agents

```python
from jselect import Index, Record, aselect, select

records = [
    {"id": "ticket-1", "text": "The verification link never arrives."},
    {"id": "ticket-2", "text": "The free trial requires a credit card."},
]
result = select(records, task="What blocks registration?", tokens=500)
payload = result.to_dict()   # same schema as --json

# Reuse an index in a process; no source re-reading or re-indexing on each query.
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
To use passages already retrieved by another system, pass them as records. A custom or semantic scorer
evaluates all eligible passages by default, even when the list exceeds the candidate limit. Set
`scan="shortlist"` to opt into retrieval before scoring.

Pass **`result.context`** to your agent. The full JSON includes additional metadata and is not subject
to the context token budget. Treat excerpts as source data rather than agent instructions.

## How it works and what it costs

1. Build or open a SQLite FTS5 index. Long records become overlapping, source-preserving passages.
   Exact repeated passages share one indexed text while retaining occurrence counts and up to five sources.
2. By default, visit every indexed passage for semantic/custom scoring. Split oversized passages before
   scoring and exclude excerpts already supplied in `--against`. Preflight the entire semantic scan's
   estimated cost before making calls.
3. Score relevance using small batches of Jev decisions. The question explicitly includes contradicting
   evidence. Scores are cached per endpoint, model, prompt version, task, and exact passage. Keep a
   relevance/diversity pool of up to 256 passages for final selection; this cap does not limit scan coverage.
4. Greedily balance relevance, text novelty, and passage token cost. Citation headers and separators count
   toward the budget; returned text is never generated. With `--sample representative`, steps 3 and 4
   keep no pool: every relevant passage enters a seeded random order, and only its front is held in memory.

With explicit `--scan shortlist`, retrieve up to 256 candidates before scoring. Most come from BM25 with
a text-diversity adjustment; 20% of slots are reserved for deterministic exploration in semantic mode.
This mode can miss evidence outside the shortlist. Local mode uses lexical scoring and a shortlist.

The defaults are eight passages per request, eight requests in flight, a 20-second total request deadline,
and a **$0.05 estimated spend budget**. Jev models are versioned (`jev-1.13.0` on TypeSafe and
`typesafe/jev-1.13` on OpenRouter); responses report the served model when available. Requests retry
transient errors within the deadline. Errors retain completed cache entries for the next run.

The dollar guard uses a conservative UTF-8 byte estimate and the listed Jev input price. Actual provider
or gateway pricing and unreported usage can differ; this is not a provider-enforced billing cap. JSON
reports measured cost when available and marks list-price estimates. Failed requests without usage
reports may have incurred additional costs. Use `--budget` to bound estimated semantic spend and
`--batch-size` to bound each request. To cap the number of passages evaluated, explicitly use
`--scan shortlist --candidates N`.
`--no-cache` disables the disk score cache at `~/.cache/jselect/scores.sqlite`.

Token counting uses `o200k_base` by default. Set `--encoding` to another tiktoken encoding/model, or `bytes`
for a conservative count with no tokenizer download. The first use of a tiktoken encoding may download its
public vocabulary. Choose the encoding your downstream model uses and leave room for its other messages.

## Measured results

Measured locally on 2026-09-19; details and frozen reports are in [the benchmark report](https://github.com/keltokhy/jselect/blob/main/docs/BENCHMARKS.md).

| Check | Observed result |
|---|---|
| One million distinct generated log records | 13.64 s to index; 0.40 s median repeated **local** query; 143 MB peak process memory |
| 30 SciFact research queries, shortlist mode, 2,000 tokens, 256 candidates, cache disabled | 86.7% mean labeled-source recall vs 78.3% for BM25 ranking; 1.86 s median; $0.149 total |
| Handwritten support, code, and contract fixtures | All 4 intended evidence types in 4 passages in each fixture; relevance-only variant covered 3, 3, and 1 |

The SciFact result measures the explicit shortlist mode, not the full-scan default. These are scoped
measurements, not guarantees for arbitrary data, agent answer quality, or future API
latency. A set selected by the default rule cannot establish prevalence or causation; for prevalence
among relevant passages, use [a representative sample](#representative-samples), which has not been
benchmarked. Diversity is a lexical heuristic; it does not certify balanced viewpoints or find every contradiction. Semantic scores are model judgments,
not calibrated confidence in a final answer. The JSON reports how much of the collection was considered.

## Output and errors

Default stdout is the exact context string. `--json` returns one object with `schema_version: 1`,
`task`, `context`, `items`, `tokens`, `token_budget`, `encoding`, `stats`, and `warnings`.
Each item contains original `text`, a stable content-hash `id`, `sources`, `occurrences`, `relevance`,
`novelty`, and the selection rule used; representative samples add `draws` and leave `novelty` null. See [the output contract](https://github.com/keltokhy/jselect/blob/main/docs/OUTPUT.md).

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

See [the validation plan](https://github.com/keltokhy/jselect/blob/main/docs/PLAN.md),
[benchmarks](https://github.com/keltokhy/jselect/blob/main/docs/BENCHMARKS.md), and the
[companion agent skill](https://github.com/keltokhy/jselect/blob/main/skills/jselect/SKILL.md). MIT licensed.
