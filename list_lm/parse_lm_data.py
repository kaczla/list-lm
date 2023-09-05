import json
import logging
import re
from pathlib import Path

from list_lm.data import ModelInfo, UrlData

LOGGER = logging.getLogger(__name__)

RGX_URL_MARKDOWN = re.compile(r"^\[(?P<title>[^]]+)\]\((?P<url>.+?)\)$")


def group_into_model_info_data(text: list[str], prefix_elements_separator: str = "- ") -> list[list[str]]:
    elements_text: list[list[str]] = []
    element_text: list[str] = []
    for line in text:
        if line.startswith(prefix_elements_separator) and element_text:
            elements_text.append(element_text)
            element_text = []

        element_text.append(line)

    if element_text:
        elements_text.append(element_text)
        del element_text

    return elements_text


def parse_url_data(text: str) -> UrlData:
    match = RGX_URL_MARKDOWN.match(text)
    if not match:
        raise RuntimeError(f"Cannot parse URL data: {repr(text)}")

    title = match.group("title")
    url = match.group("url")
    return UrlData(title=title, url=url)


def parse_single_model_info(element_text: list[str]) -> ModelInfo:
    name = element_text[0].strip().lstrip("-").strip()
    data: dict = {}
    for text in element_text[1:]:
        text = text.strip().lstrip("-").strip()
        if not text:
            continue

        if ":" not in text:
            raise RuntimeError(f"Cannot parse element: {repr(text)} in {element_text}")

        key, value = text.split(":", maxsplit=1)
        key = key.lower().strip()
        value = value.strip()
        if key == "year":
            data["year"] = int(value)
        elif key in {"publication", "code", "model weights", "video"}:
            key = key.replace(" ", "_")
            data[key] = parse_url_data(value)
        else:
            raise RuntimeError(f"Cannot parse key: {repr(key)} with value: {repr(value)}")

    return ModelInfo(name=name, **data)


def parse_markdown_to_model_info_list(markdown_path: Path) -> list[ModelInfo]:
    elements_text = group_into_model_info_data(markdown_path.read_text().splitlines())
    return [parse_single_model_info(element_text) for element_text in elements_text]


def convert_model_info_markdown_to_model_info_json(markdown_path: Path, save_path: Path) -> None:
    loaded_data = parse_markdown_to_model_info_list(markdown_path)
    json_data = [data.dict() for data in loaded_data]
    save_path.write_text(json.dumps(json_data, indent=4, ensure_ascii=False))


def convert_model_info_markdown_to_model_info_json_all() -> None:
    convert_model_info_markdown_to_model_info_json(
        Path("data/readme/language_models.md"), Path("data/json/model_data_list.json")
    )


if __name__ == "__main__":
    convert_model_info_markdown_to_model_info_json_all()