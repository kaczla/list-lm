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


def merge_lm_data(input_data: list[dict[str, Any]], dry_run: bool = False, overwrite: bool = False) -> int:
    """
    Merge input data into model_data_list.json.

    Returns:
        Error count
    """
    target_path = DATA_JSON_PATH / f"{FILE_NAME_LM_DATA}.json"
    existing_list = load_base_model_list(target_path, ModelInfo)

    # Build lookup for duplicate detection
    existing_names: set[str] = {m.name for m in existing_list}
    url_to_idx: dict[str, int] = {m.publication.url: i for i, m in enumerate(existing_list)}

    added_count = 0
    replaced_count = 0
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

        # Check for duplicate URL
        if model_info.publication.url in url_to_idx:
            if overwrite:
                existing_idx = url_to_idx[model_info.publication.url]
                old_name = existing_list[existing_idx].name
                existing_list[existing_idx] = model_info
                # Update name lookup if name changed
                if old_name != model_info.name:
                    existing_names.discard(old_name)
                    existing_names.add(model_info.name)
                replaced_count += 1
                logger.info(f"[{idx + 1}] Replacing model: {model_info.name}")
            else:
                logger.warning(
                    f"[{idx + 1}] Duplicate publication URL for {model_info.name!r}: "
                    f"{model_info.publication.url} - skipping"
                )
                duplicate_count += 1
            continue

        # Check for duplicate name (different URL)
        if model_info.name in existing_names:
            logger.warning(f"[{idx + 1}] Duplicate model name: {model_info.name!r} - skipping")
            duplicate_count += 1
            continue

        # Add to new items and update lookup sets
        new_items.append(model_info)
        existing_names.add(model_info.name)
        url_to_idx[model_info.publication.url] = -1  # Mark as seen (index not needed for new items)
        added_count += 1
        logger.info(f"[{idx + 1}] Adding model: {model_info.name}")

    has_changes = new_items or replaced_count > 0
    if has_changes and not dry_run:
        merged_list = existing_list + new_items
        logger.info(f"Saving {len(merged_list)} models to: {target_path}")
        save_base_model_list(target_path, merged_list, sort_fn=get_model_info_sort_key)  # type: ignore[arg-type]
        generate_lm_data()
        logger.info("Generated updated README")

    logger.info(
        f"Summary: {added_count} added, {replaced_count} replaced, "
        f"{duplicate_count} duplicates skipped, {error_count} errors"
    )
    changes_count = added_count + replaced_count
    if dry_run and changes_count > 0:
        logger.info(f"Run without --dry-run to merge {changes_count} items")

    return error_count


def merge_links(input_data: list[dict[str, Any]], dry_run: bool = False, overwrite: bool = False) -> int:
    """
    Merge input data into all_links.json.

    Returns:
        Error count
    """
    target_path = DATA_JSON_PATH / f"{FILE_NAME_LINKS}.json"
    existing_list = load_base_model_list(target_path, ApplicationData)

    # Build lookup for duplicate detection (name + url combination)
    key_to_idx: dict[tuple[str, str], int] = {(a.name.lower(), a.url): i for i, a in enumerate(existing_list)}

    added_count = 0
    replaced_count = 0
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
        if key in key_to_idx:
            if overwrite:
                existing_idx = key_to_idx[key]
                existing_list[existing_idx] = app_data
                replaced_count += 1
                logger.info(f"[{idx + 1}] Replacing link: {app_data.name} ({app_data.link_type})")
            else:
                logger.warning(f"[{idx + 1}] Duplicate link: {app_data.name!r} ({app_data.url}) - skipping")
                duplicate_count += 1
            continue

        # Add to new items and update lookup
        new_items.append(app_data)
        key_to_idx[key] = -1  # Mark as seen
        added_count += 1
        logger.info(f"[{idx + 1}] Adding link: {app_data.name} ({app_data.link_type})")

    has_changes = new_items or replaced_count > 0
    if has_changes and not dry_run:
        merged_list = existing_list + new_items
        logger.info(f"Saving {len(merged_list)} links to: {target_path}")
        save_base_model_list(target_path, merged_list, sort_fn=get_application_data_sort_key)
        generate_links_all()
        logger.info("Generated updated README files")

    logger.info(
        f"Summary: {added_count} added, {replaced_count} replaced, "
        f"{duplicate_count} duplicates skipped, {error_count} errors"
    )
    changes_count = added_count + replaced_count
    if dry_run and changes_count > 0:
        logger.info(f"Run without --dry-run to merge {changes_count} items")

    return error_count


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

  # Overwrite existing entries instead of skipping
  uv run python -m list_lm.merge_json lm updated_models.json --overwrite
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing entries instead of skipping duplicates",
    )

    args = parser.parse_args()
    init_logs()

    input_data = load_input_json(args.input_file)
    logger.info(f"Loaded {len(input_data)} items from: {args.input_file}")

    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")

    if args.target == MergeTarget.LM_DATA:
        errors = merge_lm_data(input_data, dry_run=args.dry_run, overwrite=args.overwrite)
    else:
        errors = merge_links(input_data, dry_run=args.dry_run, overwrite=args.overwrite)

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
