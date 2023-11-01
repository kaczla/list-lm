import logging
import re
from pathlib import Path

from list_lm.data import ApplicationData, LinkType, get_application_data_sort_key
from list_lm.data_utils import save_base_model_list

LOGGER = logging.getLogger(__name__)

RGX_LINE_LINK = re.compile(r"^\[(?P<name>[^]]+)\]\((?P<url>.+?)\)\s*[-]\s*(?P<description>.*?)$")


MAP_LINK_TYPE_NAME_TO_NORMALISED_NAME = {
    LinkType.MODEL: "model_links",
    LinkType.UTILS: "utils_links",
    LinkType.GPU_PROFILING: "gpu_profiling_links",
    LinkType.VISUALIZATION: "visualization_links",
    LinkType.VOCABULARY: "vocabulary_links",
    LinkType.OPTIMIZER: "optimizer_links",
    LinkType.DATASET: "dataset_links",
    LinkType.DOCUMENTATION: "documentation_links",
}

FILE_NAME_LINKS = "all_links"


def parse_markdown_to_data(markdown_path: Path) -> list[ApplicationData]:
    loaded_data: list[ApplicationData] = []
    file_type_name = markdown_path.stem
    with markdown_path.open("rt") as f_read:
        for line in f_read:
            line = line.strip().strip("-").strip()
            rgx_match = RGX_LINE_LINK.search(line)
            if not rgx_match:
                LOGGER.error(f"Cannot parse line with Markdown link: {repr(line)} - skipping")
                continue

            application_data = ApplicationData(
                name=rgx_match.group("name"),
                url=rgx_match.group("url"),
                description=rgx_match.group("description"),
                type_name=file_type_name,
            )
            loaded_data.append(application_data)

    return loaded_data


def convert_link_markdown_to_link_json_all() -> None:
    all_links: list[ApplicationData] = []
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
        application_data = parse_markdown_to_data(Path(f"data/readme/{file_name}.md"))
        all_links.extend(application_data)

    save_path = Path(f"data/json/{FILE_NAME_LINKS}.json")
    save_base_model_list(save_path, all_links, sort_fn=get_application_data_sort_key)


if __name__ == "__main__":
    convert_link_markdown_to_link_json_all()
