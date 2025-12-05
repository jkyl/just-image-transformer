.PHONY: all lint typecheck format check ci-lint ci-check ci

all: lint typecheck

format:
	uvx ruff format

check:
	uvx ruff check --fix

lint: format check

typecheck:
	uvx ty check

ci-format:
	uvx ruff format --check

ci-check:
	uvx ruff check

ci-lint: ci-format ci-check

ci: ci-lint typecheck
