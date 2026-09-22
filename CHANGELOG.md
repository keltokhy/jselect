# Changelog

## 0.2.0

- Score passages through the shared `jevkit-runtime` 0.2. Relevance decisions are stored per passage and
  task in the shared answer cache at `~/.cache/jev/answers.sqlite` instead of a separate score database;
  earlier score caches are not read. Provider error bodies are still withheld from output.
- Add `--sample representative` (`sample="representative"` in Python) with `--seed`: a random draw of relevant
  passage occurrences without replacement, reporting population and sample sizes for measurement use.
  Default selection and its output are unchanged.
- Add `--per FIELD` (`select_per`, `aselect_per`, `Index.select_per`): one shared scan that scores each
  distinct text once, then a separately budgeted context for each value of a field, written as JSON Lines
  in order of first appearance. Values are compared as text. Works with both selection rules.
- Make representative fitting and stopping independent of filenames, record positions, and citation
  lengths by budgeting content-ID headers. Preserve location headers when they fit that allowance and
  keep full source references in JSON. Reserve draw counts before splitting and scoring passages.
- Surface sampling and per-group warnings on stderr even with JSON output, label local sample
  populations as keyword-matching, and give full-population scans supported budget-refusal advice.
- Return an empty context per value for `--tokens 0 --per`, and reject CLI `--seed` without `--sample`.
- Clarify reproducibility conditions, the budget-dependent passage population, the limits of `--against`
  follow-ups and budget stopping, and what to check and report when using a sample in a paper.

## 0.1.1

- Default semantic and custom selection to full scans so lexical retrieval does not silently exclude relevant evidence.
- Keep lexical prefiltering available as an explicit option, with updated CLI help, documentation, and regression tests.

## 0.1.0

- Initial release of source-linked evidence selection within a token budget.
