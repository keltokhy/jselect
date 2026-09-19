.PHONY: install-local check
install-local:
	uv tool install --editable .

check:
	uv run ruff check src tests bench
	uv run ruff format --check src tests bench
	uv run pytest -q
	uv build
