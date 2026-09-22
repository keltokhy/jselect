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

# Related turns stored as separate, possibly interleaved rows, merged into one record per conversation
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
jselect "Complaints about account access" complaints.jsonl --mode semantic \
  --sample representative --seed 7 --tokens 4000 --json --stats
```

Use `--mode semantic` for a relevance-defined population: missing credentials or a failed scorer
produce an error. The default `--mode auto` uses local matching when no key is configured; warnings
reach stderr even with `--json`. `--seed` without `--sample` is a usage error.

1. The population is every passage judged relevant: relevance at or above `--threshold`, inclusive,
   which defaults to 0.5 in this mode. The scan is always complete, so `--scan shortlist` is refused;
   the dollar guard and preflight work as usual. Excerpts supplied with `--against` leave the population.
2. Passage occurrences are drawn uniformly at random, without replacement, in an order fixed by `--seed`
   (default 0). An exact repeated passage is indexed once, so a passage seen 50 times has 50 chances.
   It appears once in the context with the number of its occurrences that were drawn: `"draws":3` in
   the citation header and `draws` in JSON.
3. The draw ends at the first passage that does not fit the remaining token budget, at `-n` passages,
   or when the population runs out. Nothing is skipped to make room, because filling the gap with
   whatever fits would favor short passages. "Fits" is checked with room reserved for the largest
   draw count each repeated text could show, not the count it ends up with. Fitting reserves that
   count before scoring. Citation costs use content-ID headers so filenames and source positions
   cannot move the stopping point. The output uses location citations if they fit within that
   allowance, otherwise `{"passage":"SHA-256"}` headers; JSON `items[].sources` retains the locations.
   These reserves can leave unused tokens. Items keep the order of the draw, which carries no ranking.
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
  To avoid this stopping effect, cap displayed texts with `-n` and give `--tokens` enough room that
  `stats.sample_stop` is `max_items` rather than `budget`.
- Further draws of a text already in the context add almost no tokens and never end the draw, so a
  heavily repeated text is drawn slightly more often than its share of occurrences.
- The reserved draw count makes a repeated text cost more than it finally uses. Oversized passages
  are split with this reserve included. Any remaining empty draw caused by the reserve carries a
  specific warning. Give `--tokens` room for several passages when measuring.
- "Relevant" is the scorer's judgment at the threshold; scoring errors move the population. With
  `--local` no model judges relevance: the population is every passage sharing at least one task term
  after stop-word removal and stemming, and `--threshold` (default 0) applies to the normalized
  lexical score on top of that match. Report that as a keyword match.
- With the same multiset of parsed record texts and multiplicities, group assignments, task, seed,
  `--tokens`, `-n`, threshold, encoding, chunk size, overlap, and `--against` texts, the ordered excerpt
  texts, IDs, occurrence counts, draws, and population/sample counts repeat bit for bit, provided the
  relevance scores and jselect/tokenizer versions are unchanged. Renaming a file, using a relative
  rather than absolute path, reordering records, or moving repeated records does not change that sample.
  Source references, rendered citations, token usage, timings, and the first-appearance order of `--per`
  lines may differ. Keep the scorer/model fixed too; model responses or a custom scorer that depends on
  metadata or batch order can change the scores. `--group-by` preserves row order within each merged
  record, so reordering those rows changes the text and falls outside this guarantee.
- `--against` removes returned texts and all their occurrences from the next population. Run 1 plus
  run 2 is **not a larger representative sample**; do not report "n1+n2 of N" from the two runs. To
  enlarge a sample, rerun with a larger `--tokens` or `-n` and the same seed. The prefix is preserved
  and draws never fall while the fitted population and scores stay the same. If a budget change
  splits passages differently, it changes the population and this nesting guarantee does not apply.
- A small sample is noisy, and a representative sample still cannot establish causation. `stats`
  reports both sizes so that uncertainty can be stated.

### Using this in a paper

This is a sequential sample with a length-dependent stopping time, not a fixed-n simple random sample.
For publication, use `-n` with a roomy `--tokens` and check `stats.sample_stop == "max_items"` on every
line. `-n` caps distinct displayed texts; repeated occurrences can make the draw count larger. Report
n (`sample_occurrences`), N (`population_occurrences`), threshold, seed, scorer/model, and that the unit
is a passage occurrence. Weights live in a `"draws":N` header when more than one occurrence was drawn
(and in JSON `draws` for every item). Whether a downstream ranker honors those weights is untested.

## One context per group

`--group-by` merges related rows into one record. `--per FIELD` does the opposite job: it keeps records
as they are and returns a separate budgeted context for each value of a field, such as one dossier per
company-year.

```bash
jselect "Complaints about account access" complaints.jsonl --per company_year \
  --mode semantic --sample representative --seed 7 --tokens 2000 --json --output dossiers.jsonl

# Or save the index once; the field is recorded when the index is built
jselect index complaints.jsonl --per company_year --output complaints.jselect
jselect "Complaints about account access" complaints.jselect --per company_year --mode semantic --tokens 2000 --json
```

```python
from jselect import select_per

dossiers = select_per(records, task="Complaints about account access", per="company_year", tokens=2000)
for dossier in dossiers:
    print(dossier.per["value"], dossier.tokens, len(dossier.items))
```

Use `--mode semantic` when each dossier must be selected by semantic relevance. Otherwise `auto`
can use keyword matching when no key is configured; local-mode warnings also appear on stderr with
JSON Lines output.

Dictionaries and rows supply the field directly; a `Record` supplies it in `metadata`, for example
`Record(text, metadata={"company_year": "acme-2021"})`.

The collection is scanned once and each distinct passage text is scored once, however many groups it
appears in. Selection then runs separately for each value with its own `--tokens` budget, under either
rule, as if that group were the whole collection: occurrence counts, citations, and sample populations
are the group's own. A representative sample of a group is the same one that selecting that group alone
with the same seed would give; a text shared by several groups gets the same random number in each, so
samples of different groups are not independent on shared text. (With `--local`, lexical scores are
still computed over the whole collection.) Output is JSON Lines, so `--json` is required: one object per value,
in order of first appearance, in the usual schema plus `per` (`field`, `value`, and the value's indexed
`passages` and `occurrences`). A value with no relevant passage still gets a line with an empty context.
With `--tokens 0`, every value gets an empty context object without scoring.
`--against` accepts the JSON Lines of an earlier `--per` run; an excerpt it lists is excluded from every group.

Limits: the field must hold a string or integer in every record, and one field is supported, so build
a combined column for keys such as company and year. Values are compared as text, so 1 and "1" name the
same group, as with `--group-by`. With `--group-by`, all rows of a merged record must share the value. `--scan shortlist` is refused, and `--local` reads every lexical match instead of a
shortlist, because one global shortlist would starve small groups. `calls`, `cost`, and timings in each
line describe the shared scan; do not sum them across lines. Memory grows with the number of groups,
most under the default rule, which keeps up to `--candidates` passages for each, and with the collection:
to score a text once, a scan under `--per` or `--sample` remembers one score per distinct passage (about
150 bytes each). Excerpts subdivided to fit a budget smaller than one passage can differ between groups;
different excerpts are scored separately, identical ones once.

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
`novelty`, and the selection rule used; representative samples add `draws` and leave `novelty` null.
With `--per`, stdout is one JSON object per line, each with a `per` key. See [the output contract](https://github.com/keltokhy/jselect/blob/main/docs/OUTPUT.md).

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

See [benchmark methods and results](https://github.com/keltokhy/jselect/blob/main/docs/BENCHMARKS.md) and the
[companion agent skill](https://github.com/keltokhy/jselect/blob/main/skills/jselect/SKILL.md). MIT licensed.

## Shared JevKit development

This development branch uses `jevkit-core>=0.1.0,<0.2.0`. Clone the core beside this
repository as `../jevkit-core`; `uv sync` installs it editably. Core Python edits
then apply on the next run of this tool. Restart long-lived Python processes.
The core is not yet published; this branch requires the sibling checkout until
its initial release is available.

From the core checkout, `python scripts/dev.py setup`, `check`, and `wheel-check`
set up and validate the five consumers in separate environments. For the current
isolated migration worktrees, add `--suffix=-jevkit` before the subcommand.
The shared core README documents compatibility and release sequencing.

CLI behavior and product algorithms stay in this repository. Consumer CI expects
the core remote's `v0.1.0` tag; publish the core before publishing this branch.
