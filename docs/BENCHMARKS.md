# Measurements, 2026-09-19

These checks cover retrieval, selection, cost, and local throughput. They do **not** measure downstream
agent answer quality, represent all workloads, or establish universal superiority over retrieval systems.
The comparison is against the named baselines, not a trained embedding retriever or cross-encoder.

## Public research retrieval: BEIR SciFact

Source: [BEIR's SciFact distribution](https://github.com/beir-cellar/beir#beers-available-datasets).
The runner downloads the public archive, records its SHA-256, and uses its original relevance judgments.
It builds an index over titles and abstracts. Input labels are never sent to the relevance scorer.

```bash
uv run python bench/retrieval.py --queries 30 --seed 1910 --mode semantic \
  --candidates 256 --tokens 2000 --budget 0.25 --no-cache
```

| Measure | Observed |
|---|---:|
| Queries sampled, fixed seed 1910 | 30 |
| Maximum evidence tokens per query | 2,000, using o200k_base |
| Maximum candidates per query | 256 |
| Mean fraction of gold source documents present in shortlist | 96.7% |
| Mean gold-document recall of BM25-ranked context | 78.3% |
| Mean gold-document recall of jselect context | 86.7% |
| Median jselect query time, model cache disabled | 1.86 s |
| Provider-reported cost for all 30 queries | $0.149183 |
| Mean cost per query | $0.004973 |

Both methods pack source excerpts and identical citation headers into the same token budget. BM25 uses
SQLite's porter/unicode61 FTS5 rank and takes passages in rank order while they fit. This jselect run used
shortlist candidate diversification, exploration, batched semantic relevance, and final greedy packing.
The runner explicitly keeps `scan="shortlist"` to reproduce this protocol. These measurements do not
describe the current full-scan default.
Recall is averaged across queries at the **source-document** level. Multiple excerpts from one document
do not increase recall. A retrieved document may contain a relevant fact outside its selected excerpt;
this metric alone cannot prove that an agent received every necessary fact.

The provider reported `typesafe/jev-1.13-20260917`. The request model was `typesafe/jev-1.13` on OpenRouter.
The benchmark used a 40-second request timeout for robustness; ordinary CLI runs default to 20 seconds.
Index construction is excluded from query time. Requests were uncached, but OS/disk caches were not flushed.
Results are a small fixed sample without statistical significance claims.

An earlier ten-query smoke test exposed a real failure: the initial prompt undervalued contradictory
evidence, producing 70% recall versus 100% for its lexical baseline. The revised prompt explicitly asks
for evidence that disproves assumptions. The 30-query check used a different sample. Wider shortlisting
improved candidate recall but did not improve final recall on that sample: relevance scoring still drops
some useful sources. One labeled abstract only discussed adjacent subject matter; its gold claim was not
directly stated in the abstract. Both model behavior and document-level labels limit this evaluation.

A cache-enabled 64-candidate run is retained as development evidence. Its timing and cost include cache
hits and must not be presented as cold inference performance. During development, one provider timeout
interrupted a run; completed decisions remained cached. The runner now checkpoints every completed query
and records errors and available spending stats.

Frozen result: [SciFact cold run](benchmarks/scifact-cold.json).

## Cross-domain selection fixtures

`uv run python bench/scenarios.py` creates 75 records per task: 24 near-repeated relevant passages,
three distinct relevant passages, and 48 unrelated distractors. The hand-authored facet labels are used
only for scoring the output, never for the model request. Every method gets 1,000 tokens and four passages.

| Fixture | jselect evidence types covered | Same scorer/packer, diversity disabled |
|---|---:|---:|
| Signup support: email, card requirement, SSO, successful workaround | 4/4 | 3/4 |
| Worker memory: retained jobs, unbounded cache, listener, successful cleanup | 4/4 | 3/4 |
| Subscription contract: notice, exception, fees, overriding amendment | 4/4 | 1/4 |

These deliberately test a mechanism: whether repetitive high-scoring examples crowd out distinct
information. They are not a production dataset, an independent legal/code assessment, or evidence of
general 100% accuracy. Both variants use identical cached relevance decisions. The comparison disables
only the final diversity penalty; relevance-only packing still accounts for passage token cost.

Frozen result: [selection fixtures](benchmarks/scenarios.json).

## Local scale

`uv run python bench/scale.py --records 1000000` streams one million generated short log records into a
temporary disk-backed index. Each text has a distinct ticket number; no exact duplicate removal reduces
the input. Four base templates make this a throughput test, not a semantic quality test.

| Measure | Observed |
|---|---:|
| Input characters | 71,888,890 |
| Index build time | 13.64 s |
| Median of three repeated local queries | 0.40 s |
| Index size | 544,395,264 bytes |
| Peak process resident memory | 142,999,552 bytes |
| Model calls / API cost | 0 / $0 |

Measured with CPython 3.13.15 on macOS arm64. Resident memory includes Python and the tokenizer. The first
query includes tokenizer initialization; the tokenizer vocabulary was already downloaded. Raw JSON arrays,
individual text documents, and each grouped conversation are materialized as individual records; this
measurement does not claim bounded memory for one arbitrarily large document or JSON array.

Frozen result: [million-record scale run](benchmarks/scale-million.json).

## Contract verification

The automated suite uses synthetic data and HTTP mocks; it makes no paid calls. It exercises exact token
limits, Unicode/special-token literals, source spans and hashes, grouping interleaved turns, long-document
tails, repeated sources, follow-up exclusion, custom scorers, async integration, complete-scan preflight,
bounded full-scan batches, provider context limits, malformed responses, cache isolation, timeouts, and
CLI JSON errors. Real semantic quality is assessed by the separate live checks above.

Re-run `make check` before relying on a new revision. Model/pricing changes require a new live benchmark.
