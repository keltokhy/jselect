# Changelog

## Unreleased

- Add `--sample representative` (`sample="representative"` in Python) with `--seed`: a random draw of relevant
  passage occurrences without replacement, reporting population and sample sizes for measurement use.
  Default selection and its output are unchanged.
- Add `--per FIELD` (`select_per`, `aselect_per`, `Index.select_per`): one shared scan, then a separately
  budgeted context for each value of a field, written as JSON Lines. Works with both selection rules.

## 0.1.1

- Default semantic and custom selection to full scans so lexical retrieval does not silently exclude relevant evidence.
- Keep lexical prefiltering available as an explicit option, with updated CLI help, documentation, and regression tests.

## 0.1.0

- Initial release of source-linked evidence selection within a token budget.
