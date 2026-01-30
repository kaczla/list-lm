"""Merge JSON data into LM data or links files."""

import argparse
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from list_lm.data import (
    ApplicationData,
    ModelInfo,
    get_application_data_sort_key,
    get_model_info_sort_key,
)
from list_lm.data_utils import load_base_model_list, save_base_model_list
from list_lm.generate_readme import generate_links_all, generate_lm_data
from list_lm.log_utils import init_logs
from list_lm.parse_links import FILE_NAME_LINKS
from list_lm.parse_lm_data import FILE_NAME_LM_DATA

DATA_JSON_PATH = Path("data/json")


class MergeTarget(StrEnum):
    LM_DATA = "lm"
    LINKS = "links"


def load_input_json(input_path: Path) -> list[dict[str, Any]]:
    """Load input JSON file containing list of dictionaries."""
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    try:
        data = json.loads(input_path.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        logger.error("Input JSON must be a list of dictionaries")
        sys.exit(1)

    return data


def merge_lm_data(input_data: list[dict[str, Any]], dry_run: bool = False) -> tuple[int, int, int]:
    """
    Merge input data into model_data_list.json.

    Returns:
        Tuple of (added_count, duplicate_count, error_count)
    """
    target_path = DATA_JSON_PATH / f"{FILE_NAME_LM_DATA}.json"
    existing_list = load_base_model_list(target_path, ModelInfo)

    # Build lookup sets for duplicate detection
    existing_names: set[str] = {m.name for m in existing_list}
    existing_urls: set[str] = {m.publication.url for m in existing_list}

    added_count = 0
    duplicate_count = 0
    error_count = 0
    new_items: list[ModelInfo] = []

    for idx, item in enumerate(input_data):
        try:
            model_info = ModelInfo(**item)
        except ValidationError as e:
            logger.error(f"[{idx + 1}] Invalid ModelInfo data: {e}")
            error_count += 1
            continue

        # Check for duplicates
        if model_info.name in existing_names:
            logger.warning(f"[{idx + 1}] Duplicate model name: {model_info.name!r} - skipping")
            duplicate_count += 1
            continue

        if model_info.publication.url in existing_urls:
            logger.warning(
                f"[{idx + 1}] Duplicate publication URL for {model_info.name!r}: "
                f"{model_info.publication.url} - skipping"
            )
            duplicate_count += 1
            continue

        # Add to new items and update lookup sets
        new_items.append(model_info)
        existing_names.add(model_info.name)
        existing_urls.add(model_info.publication.url)
        added_count += 1
        logger.info(f"[{idx + 1}] Adding model: {model_info.name}")

    if new_items and not dry_run:
        merged_list = existing_list + new_items
        logger.info(f"Saving {len(merged_list)} models to: {target_path}")
        save_base_model_list(target_path, merged_list, sort_fn=get_model_info_sort_key)  # type: ignore[arg-type]
        generate_lm_data()
        logger.info("Generated updated README")

    return added_count, duplicate_count, error_count


def merge_links(input_data: list[dict[str, Any]], dry_run: bool = False) -> tuple[int, int, int]:
    """
    Merge input data into all_links.json.

    Returns:
        Tuple of (added_count, duplicate_count, error_count)
    """
    target_path = DATA_JSON_PATH / f"{FILE_NAME_LINKS}.json"
    existing_list = load_base_model_list(target_path, ApplicationData)

    # Build lookup set for duplicate detection (name + url combination)
    existing_keys: set[tuple[str, str]] = {(a.name.lower(), a.url) for a in existing_list}

    added_count = 0
    duplicate_count = 0
    error_count = 0
    new_items: list[ApplicationData] = []

    for idx, item in enumerate(input_data):
        try:
            app_data = ApplicationData(**item)
        except ValidationError as e:
            logger.error(f"[{idx + 1}] Invalid ApplicationData data: {e}")
            error_count += 1
            continue

        # Check for duplicates (same name and URL)
        key = (app_data.name.lower(), app_data.url)
        if key in existing_keys:
            logger.warning(f"[{idx + 1}] Duplicate link: {app_data.name!r} ({app_data.url}) - skipping")
            duplicate_count += 1
            continue

        # Add to new items and update lookup set
        new_items.append(app_data)
        existing_keys.add(key)
        added_count += 1
        logger.info(f"[{idx + 1}] Adding link: {app_data.name} ({app_data.link_type})")

    if new_items and not dry_run:
        merged_list = existing_list + new_items
        logger.info(f"Saving {len(merged_list)} links to: {target_path}")
        save_base_model_list(target_path, merged_list, sort_fn=get_application_data_sort_key)
        generate_links_all()
        logger.info("Generated updated README files")

    return added_count, duplicate_count, error_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge JSON data into LM data or links files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge models from a JSON file
  uv run python -m list_lm.merge_json lm new_models.json

  # Merge links from a JSON file
  uv run python -m list_lm.merge_json links new_links.json

  # Dry run to preview what would be added
  uv run python -m list_lm.merge_json lm new_models.json --dry-run
        """,
    )
    parser.add_argument(
        "target",
        type=str,
        choices=[MergeTarget.LM_DATA.value, MergeTarget.LINKS.value],
        help="Target to merge into: 'lm' for model data, 'links' for application links",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to JSON file containing list of items to merge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be merged without making changes",
    )

    args = parser.parse_args()
    init_logs()

    input_data = load_input_json(args.input_file)
    logger.info(f"Loaded {len(input_data)} items from: {args.input_file}")

    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")

    if args.target == MergeTarget.LM_DATA:
        added, duplicates, errors = merge_lm_data(input_data, dry_run=args.dry_run)
    else:
        added, duplicates, errors = merge_links(input_data, dry_run=args.dry_run)

    # Summary
    logger.info("=" * 50)
    logger.info(f"Summary: {added} added, {duplicates} duplicates skipped, {errors} errors")

    if args.dry_run and added > 0:
        logger.info(f"Run without --dry-run to merge {added} items")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
