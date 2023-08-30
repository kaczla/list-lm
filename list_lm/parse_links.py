import json
import logging
import re
from pathlib import Path

from list_lm.data import ApplicationData

LOGGER = logging.getLogger(__name__)

RGX_LINE_LINK = re.compile(r"^\[(?P<name>[^]]+)\]\((?P<url>.+?)\)\s*[-]\s*(?P<description>.*?)$")


def parse_markdown_to_data(markdown_path: Path) -> list[ApplicationData]:
    loaded_data: list[ApplicationData] = []
    with markdown_path.open("rt") as f_read:
        for line in f_read:
            line = line.strip().strip("-").strip()
            rgx_match = RGX_LINE_LINK.search(line)
            if not rgx_match:
                LOGGER.error(f"Cannot parse line with Markdown link: {repr(line)} - skipping")
                continue

            application_data = ApplicationData(
                name=rgx_match.group("name"), url=rgx_match.group("url"), description=rgx_match.group("description")
            )
            loaded_data.append(application_data)

    return loaded_data


def convert_markdown_to_json(markdown_path: Path, save_path: Path) -> None:
    loaded_data = parse_markdown_to_data(markdown_path)
    json_data = sorted([data.dict() for data in loaded_data], key=lambda x: x["name"])
    save_path.write_text(json.dumps(json_data, indent=4, ensure_ascii=False))


def convert_markdown_to_json_all() -> None:
    for file_name in [
        "dataset_links",
        "documentation_links",
        "gpu_profiling_links",
        "model_links",
        "optimizer_links",
        "utils_links",
        "visualization_links",
        "vocabulary_links",
    ]:
        convert_markdown_to_json(Path(f"data/readme/{file_name}.md"), Path(f"data/json/{file_name}.json"))
