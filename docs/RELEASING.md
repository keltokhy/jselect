# Releases

The GitHub repository is https://github.com/keltokhy/jselect. The PyPI distribution is
`jev-select`; the executable and Python import are `jselect`.

## Publishing

1. Update the version in `pyproject.toml` and `src/jselect/__init__.py`, then run `uv lock`.
2. Run `make check` and `uvx --from twine==7.0.0 twine check --strict dist/*`.
3. Commit and push to `main`, then tag that commit as `vVERSION` and push the tag.
4. The `publish.yml` workflow checks Python 3.10 and 3.13 before publishing the wheel and
   source distribution through PyPI Trusted Publishing. The tag must match the package version.
5. Verify PyPI's artifact hashes, install the release from PyPI in a fresh environment outside
   the checkout, and exercise the CLI and Python interface. Create the matching GitHub release.

PyPI's publisher must name owner `keltokhy`, repository `jselect`, workflow `publish.yml`,
and environment `pypi`. No long-lived PyPI token is stored in this repository.

## 0.1.0

Initial release prepared on 2026-09-19. Publication verification will be recorded after the
registry upload and installation checks complete.
