#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "[ERROR] Usage: $0 <file.json> [extra merge_json args]" >&2
    echo "[ERROR] File name must contain 'model' (LM data) or 'link' (links data)." >&2
    exit 1
fi

file="$1"
shift

if [ ! -f "$file" ]; then
    echo "[ERROR] File not found: ${file}" >&2
    exit 1
fi

name="$(basename "$file")"
lower="$(echo "$name" | tr '[:upper:]' '[:lower:]')"

case "$lower" in
    *model*)
        data_type="lm"
        ;;
    *link*)
        data_type="links"
        ;;
    *)
        echo "[ERROR] Cannot detect data type from file name: ${name}" >&2
        echo "[ERROR] File name must contain 'model' (LM data) or 'link' (links data)." >&2
        exit 1
        ;;
esac

echo "[LOG] Detected data type '${data_type}' from file: ${name}"
echo "[LOG] Merging ${file}..."
uv run python -m list_lm.merge_json "$data_type" "$file" "$@"

echo "[LOG] Done!"
