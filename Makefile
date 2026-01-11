all: format lint_fix lint_shell
	@echo "[INFO] All checks complete!"

lint:
	@echo "[INFO] Running lint..."
	@uv run ruff check list_lm

lint_fix:
	@echo "[INFO] Running lint fix..."
	@uv run ruff check --select I --fix list_lm
	@uv run ruff check --fix list_lm

lint_shell:
	@echo "[INFO] Running shell check..."
	@uv run shellcheck scripts/*.sh

format:
	@echo "[INFO] Running format..."
	@uv run ruff format list_lm

format_toml:
	@echo "[INFO] Running TOML format..."
	@taplo fmt pyproject.toml

type_check:
	@echo "[INFO] Running type check..."
	@uv run mypy list_lm

.PHONY: lint lint_fix format format_toml type_check
