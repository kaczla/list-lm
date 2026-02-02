#!/usr/bin/env bash

echo "[LOG] Validate all models and links..."
uv run -m list_lm.validate_all

echo "[LOG] Generate README files..."
uv run -m list_lm.generate_readme

echo "[LOG] Done!"
