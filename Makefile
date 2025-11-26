.PHONY: all lint typecheck format check

all: lint typecheck

format:
	uvx ruff format

check:
	uvx ruff check --fix

lint: format check

typecheck:
	uvx ty check
