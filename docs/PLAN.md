# Product contract and validation

Goal: a separate, broadly useful tool that is easy to use, powerful, inexpensive, and fast.
No jgrep dependency and no changes to its repository.

## User contract

- `jselect "task or question" files... --tokens 8000` and `select(records, task=..., tokens=8000)`.
- Text, code, logs, JSONL, JSON, CSV, directories, standard input, and Python record iterables.
- Source-faithful excerpts with stable identifiers, source locations, and exact final-context token accounting.
- Automatic bounded local retrieval and semantic relevance scoring, followed by selection that penalizes
  repetition. No manual labels or taxonomy. Existing retrieved passages and custom scorers also work.
- `--against` excludes exact previous excerpts and discourages similar material on follow-up questions.
- Local mode requires no credentials or paid requests. Semantic mode uses configured Jev credentials,
  versioned prompts/models, bounded concurrency, retries, persistent caching, and explicit cost accounting.
- Persisted SQLite indexes make repeated queries cheap. Broad semantic scanning is opt-in and bounded by cost.
- JSON output has a stable versioned schema, separate context and metadata, coverage and limitations.
- Empty evidence is a valid result. Bad data, malformed scores, and provider errors are explicit failures.

## Evidence required before completion

1. CLI and Python examples work outside this source directory after installation.
2. Automated contracts for provenance, Unicode, budget boundaries, all input adapters, batching/caching,
   malformed responses, failures, custom scorers, deduplication, and novelty against prior evidence.
3. Existing jgrep repository remains unchanged.
4. Selection evaluated against relevance-only and lexical baselines under the same token budget on
   multiple tasks, including non-overlapping wording, contradictory evidence, and redundant distractors.
5. A public labeled retrieval dataset tests real semantic quality and shortlist recall. Report negative results.
6. Large-input benchmark reports build time, query latency, memory, and cold/warm model cost separately.
7. Usage docs explain source boundaries, model limitations, selection bias, tokenization, and actual measurements.

Quality claims must match evidence. Diversity is a heuristic, not a guarantee of completeness, balanced
representation, statistical prevalence, causality, or finding every contradiction.

## Completion evidence

Validated on 2026-09-19:

| Requirement | Authoritative evidence |
|---|---|
| Standalone tool and library, isolated from jgrep | `pyproject.toml`, `src/jselect`, separate Git repository; jgrep retains only its pre-existing untracked `.DS_Store` |
| Simple CLI and Python interface, available outside checkout | Installed `jselect` on PATH; clean-wheel subprocess tests in a temporary directory; `docs/verification.json` |
| Multiple input types, grouping, indexes, reuse, and custom async/sync scorers | `tests/test_selection.py`, `tests/test_streaming_groups.py`, `tests/test_judge_cli.py` |
| Strict output budget and source fidelity | Unicode and tokenizer matrix, dense emoji fitting, hashes/spans, parent lookup, source grouping, special-token literal tests |
| Bounded model work and transparent failures | Full-scan preflight, batch and provider-context bounds, response validation, timeout/retry, JSON error and cache isolation tests |
| Useful selection across contexts | Three live support/code/contract fixtures; distinct-evidence coverage compared with the same scorer without the diversity penalty |
| Public labeled retrieval and baseline | Frozen 30-query SciFact check, equal 2,000-token budget; 86.7% vs 78.3% labeled-document recall; errors and caveats in `docs/BENCHMARKS.md` |
| Fast and inexpensive operation | One-million-record local run; uncached semantic cost/latency report; installed warm replay with zero paid calls |
| Packaging, documentation, agent use | Wheel and sdist built; `README.md`, `docs/OUTPUT.md`, validated companion skill installed by symlink |

Remaining product limits are explicit: shortlist recall can fail, novelty is lexical, provider latency and
pricing vary, and selection quality is not a proof of downstream answer quality. There is no claim of
production validation or universal superiority. The initial implementation was validated before release;
publication is tracked separately in `docs/RELEASING.md`.
