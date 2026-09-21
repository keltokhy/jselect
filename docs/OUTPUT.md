# Output contract, schema version 1

`jselect "question" data.jsonl --json` and `Selection.to_dict()` use the same schema.
Additional fields may be added within version 1. Consumers should ignore fields they do not recognize.
Representative sampling was added this way: `schema_version` stays 1, its fields appear only when
`--sample representative` is requested, and default output is unchanged.

```json
{
  "schema_version": 1,
  "task": "What blocks signup?",
  "context": "[1] {\"source\":\"tickets.jsonl\",\"line\":2,\"id\":\"t2\",\"chars\":[0,19],\"field\":\"text\"}\nEmail never arrives",
  "tokens": 50,
  "token_budget": 1000,
  "encoding": "o200k_base",
  "items": [],
  "stats": {"mode": "semantic", "calls": 1, "cost": 0.00001},
  "warnings": []
}
```

The example illustrates the envelope; actual token counts and item arrays are computed from the result.

## Evidence

`context` is the complete bounded string, including source headers and separators. `tokens` counts
that string with the named encoding. It does not count `task`, the JSON envelope, or another model's
messages. Empty context uses zero tokens. No generated summary is substituted for source text.

An item has:

- `id`: SHA-256 of the exact excerpt text. Equal excerpt texts share an ID even across sources.
- `text`: original excerpt, or an exact substring of the canonical text representation described below.
- `sources`: up to five original locations for exactly repeated text. The first appears in context.
- `occurrences`: exact repeated passage occurrences in the indexed snapshot, not a frequency estimate
  for an issue or a count of independently affected users.
- `relevance`: normalized lexical score, a Jev decision, or the custom scorer's number in [0,1].
- `novelty`: one minus maximum weighted lexical similarity to previously supplied/selected excerpts.
  `null` in a representative sample, where novelty plays no part.
- `reason`: the actual algorithmic selection rule; it is not a generated explanation of the evidence.
- `draws`: representative samples only. How many of this text's `occurrences` fell in the sample,
  from 1 to `occurrences`. A value above 1 also appears as `draws` in the item's context header.

Each source contains `source`, `record_id`, one-based `line`, `end_line`, zero-based Unicode character
`start` and exclusive `end`, `field`, `structured`, and `record_sha256` of the full canonical record text.
`passage_id` identifies the indexed parent and can be passed to `show` even when the output excerpt was
subdivided to fit a small budget. Offsets count characters, not bytes or tokens. Fitted excerpts inherit
the parent passage's duplicate count; this is not an exhaustive count of the shorter text in the corpus.

For text/code files, offsets are relative to the complete file decoded with UTF-8-SIG (initial BOM removed),
with original line endings preserved. For lines/JSONL/CSV, they are
relative to the selected record text or decoded field. A structured row's `line` is its physical start
line; embedded newlines in a field do not change it. JSON arrays use `json_index` and file line 1;
do not interpret an array item's position as a physical line number.

String fields are retained exactly. Nonstring fields and whole JSON objects are serialized using
`json.dumps(value, ensure_ascii=False, sort_keys=True)`. Their offsets refer to that canonical representation,
not to the whitespace or property ordering in the source file. The hash allows callers to verify the
representation before applying offsets.

Grouped rows use a canonical concatenation of member texts separated by two newlines. Their references
include `group_by`, `group_id`, and overlapping `members` with original file/row/field locations and
character spans in the group's canonical text. Input order is preserved. Group context headers list the
source rows; detailed member mappings are available in JSON.

## Representative samples

With `--sample representative`, items are a seeded random sample instead of a relevance/novelty
selection, in the order drawn. The population is every fitted, previously unseen passage with relevance
at or above `threshold` (default 0.5; in local mode every passage sharing a task term, default 0).
Each occurrence of each such passage receives a uniform random key derived from the seed, the passage
ID, and its first source location. The sample is every occurrence whose key is below a cutoff, and the
cutoff is the key of the first passage that cannot be added: it does not fit the remaining token budget
or would exceed `--max-items`. A repeated text is shown once with its `draws`. The fit check reserves
room for the largest `draws` value a text could show, so a context can end a few tokens short.

`stats` then has `selection_method: "random_occurrence_sample_without_replacement"` and:

- `sample`: `"representative"`; `seed`: the seed used; `threshold`: the population's relevance floor.
- `population_passages`, `population_occurrences`: distinct relevant passages and their total occurrences.
  Identical excerpts produced by subdividing different parents can be counted as separate passages
  here; if drawn, they are merged into one item.
- `sample_passages`, `sample_occurrences`: items returned and the sum of their `draws`.
- `sample_stop`: what ended the draw: `budget`, `max_items`, or `population` (every occurrence was drawn).
- `candidates` equals `population_passages`. `candidate_limit` and `diversity` are absent because
  neither applies. `scan` is `all`, including in local mode, where the lexical index is read in full.

The passage that ends the draw is excluded and is more often long, so long passages are somewhat
under-represented unless `--max-items` ends the draw first. Repeated draws of an included text add
almost no tokens, so heavily repeated texts are drawn slightly more often than their share. The unit
is a passage occurrence; records that span several passages have proportionally more chances.

## One context per group

`--per FIELD --json` and `select_per(...)` return one object per value of the field, as JSON Lines in
order of first appearance. Each line is a complete object in this schema with one more key:

```json
{"per": {"field": "company_year", "value": "acme-2021", "passages": 61, "occurrences": 80}}
```

The values shown are illustrative. `passages` and `occurrences` count the value's distinct indexed passages and their occurrences, relevant
or not, before any fitting or `--against` exclusion. Within a line, `occurrences`, `sources`, `draws`,
`candidates`, `selected`, and the population and sample counts belong to that value alone, and every
source carries `per` with the value. Index counts, `passages_evaluated`, `scored_passages`, `calls`,
`cost`, and the timings other than `selection_seconds` describe the single shared scan and repeat on
every line. A value with no relevant passage has an empty `context` and the usual warning. An index
built with `--per` adds `per` and `per_values` to its `stats`; it remains a schema version 1 index and
still answers ordinary queries. An error replaces the whole stream with the single error object.

## Statistics and limits

- `records`, `passages`, `unique_passages`, `characters`: indexed snapshot counts. Grouped conversations
  count as records. Long records can create multiple overlapping passages.
- `source_passages_considered`: indexed passages entering retrieval/preparation.
- `scan`: resolved scan mode, `all` or `shortlist`. Defaults to `all` for semantic/custom scoring and
  `shortlist` for local mode.
- `passages_evaluated`: fitted passages given to the local/semantic/custom scorer.
- `candidates`: passages retained for final packing; capped by `candidate_limit`.
- `scored_passages`: newly scored by Jev; `cached_passages` were reused. Neither measures correctness.
- `calls`, `retries`, `input_tokens`, `cost`, `cost_source`, `model`, `api`: provider work and provenance.
- `index_seconds`, `retrieval_seconds`, `scoring_seconds`, `selection_seconds`, `seconds`: wall-clock stages.

Token fitting can subdivide candidates. In the default semantic/custom full-scan mode, every fitted,
previously unseen passage is evaluated, and a relevance/diversity pool bounded by `--candidates` is retained.
The pool and final packing are heuristic. The candidate cap limits retained passages, not full-scan
evaluation coverage. A semantic
scan that exceeds the estimated dollar budget fails before scoring requests; it does not fall back to a
shortlist. Local mode uses lexical matching; explicit semantic shortlist mode cannot recover a passage
that never enters its shortlist. Warnings disclose these boundaries.

Source texts may contain instructions or misleading claims. The selector's prompt tells its scorer to
treat text as evidence. This does not make excerpts trusted instructions for a consuming agent.

## Other commands

`doctor --json` reports local readiness, semantic configuration, auth source category, and model.
It does not contact the provider or verify the credential. `index --json` and `inspect --json` return
`schema_version`, saved `path`, and index `stats`. `show --json` returns `schema_version` and `passage`.

Errors return exit 2 and:

```json
{"schema_version":1,"error":{"type":"SemanticError","message":"estimated spend exceeds budget"},"stats":null}
```

When available, error stats include work completed before failure. Cached successful decisions remain
usable on a rerun. No partial evidence bundle is presented as a successful result.
