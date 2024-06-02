import logging
import re
from pathlib import Path

from list_lm.data import ArticleData, ModelInfo, UrlData, get_model_info_sort_key
from list_lm.data_utils import save_base_model_list
from list_lm.utils import convert_string_to_date

LOGGER = logging.getLogger(__name__)

RGX_URL_MARKDOWN = re.compile(r"^\[(?P<title>[^]]+)\]\((?P<url>.+?)\)$")
RGX_URL_MARKDOWN_WITH_DATE = re.compile(
    r"^\[(?P<title>[^]]+)\]\((?P<url>.+?)\)\s+\((?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\)$"
)

FILE_NAME_LM_DATA = "model_data_list"
FILE_NAME_LM_MARKDOWN = "language_models"


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


def parse_article_data(text: str) -> ArticleData:
    match = RGX_URL_MARKDOWN_WITH_DATE.match(text)
    if not match:
        raise RuntimeError(f"Cannot parse article data: {repr(text)}")

    title = match.group("title")
    url = match.group("url")
    date = match.group("date")
    return ArticleData(title=title, url=url, date_create=convert_string_to_date(date))


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
        elif key in {"publication"}:
            key = key.replace(" ", "_")
            data[key] = parse_article_data(value)
        elif key in {"code", "model weights", "video"}:
            key = key.replace(" ", "_")
            if key.startswith("model_"):
                key = key[6:]
            data[key] = parse_url_data(value)
        else:
            raise RuntimeError(f"Cannot parse key: {repr(key)} with value: {repr(value)}")

    return ModelInfo(name=name, **data)


def parse_markdown_to_model_info_list(markdown_path: Path) -> list[ModelInfo]:
    elements_text = group_into_model_info_data(markdown_path.read_text().splitlines())
    return [parse_single_model_info(element_text) for element_text in elements_text]


def convert_model_info_markdown_to_model_info_json(markdown_path: Path, save_path: Path) -> None:
    loaded_data = parse_markdown_to_model_info_list(markdown_path)
    save_base_model_list(save_path, loaded_data, sort_fn=get_model_info_sort_key)  # type: ignore[arg-type]


def convert_model_info_markdown_to_model_info_json_all() -> None:
    convert_model_info_markdown_to_model_info_json(
        Path(f"data/readme/{FILE_NAME_LM_MARKDOWN}.md"), Path(f"data/json/{FILE_NAME_LM_DATA}.json")
    )


if __name__ == "__main__":
    convert_model_info_markdown_to_model_info_json_all()
