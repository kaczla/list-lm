import json
import logging
import re
from pathlib import Path

from list_lm.data import ApplicationData, LinkType

LOGGER = logging.getLogger(__name__)

RGX_LINE_LINK = re.compile(r"^\[(?P<name>[^]]+)\]\((?P<url>.+?)\)\s*[-]\s*(?P<description>.*?)$")


MAP_LINK_TYPE_NAME_TO_FILE_NAME = {
    LinkType.MODEL: "model_links",
    LinkType.UTILS: "utils_links",
    LinkType.GPU_PROFILING: "gpu_profiling_links",
    LinkType.VISUALIZATION: "visualization_links",
    LinkType.VOCABULARY: "vocabulary_links",
    LinkType.OPTIMIZER: "optimizer_links",
    LinkType.DATASET: "dataset_links",
    LinkType.DOCUMENTATION: "documentation_links",
}


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


def convert_link_markdown_to_link_json(markdown_path: Path, save_path: Path) -> None:
    loaded_data = parse_markdown_to_data(markdown_path)
    json_data = sorted([data.model_dump(mode="json") for data in loaded_data], key=lambda x: x["name"].lower())
    save_path.write_text(json.dumps(json_data, indent=4, ensure_ascii=False))


def convert_link_markdown_to_link_json_all() -> None:
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
        convert_link_markdown_to_link_json(Path(f"data/readme/{file_name}.md"), Path(f"data/json/{file_name}.json"))


def convert_link_type_to_file_name(link_type: LinkType) -> str:
    if link_type in MAP_LINK_TYPE_NAME_TO_FILE_NAME:
        return MAP_LINK_TYPE_NAME_TO_FILE_NAME[link_type]

    raise ValueError(f"Cannot get file path to link type: {link_type.name}")


if __name__ == "__main__":
    convert_link_markdown_to_link_json_all()
