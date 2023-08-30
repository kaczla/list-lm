import json
import re
from pathlib import Path

from list_lm.data import ModelInfo, UrlData

RGX_URL_MARKDOWN = re.compile(r"^\[(?P<title>[^]]+)\]\((?P<url>.+?)\)$")


def group_elements(text: list[str], prefix_elements_separator: str = "- ") -> list[list[str]]:
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


def read_section_elements(readme_file_path: Path, section_name: str) -> list[list[str]]:
    found_section = False
    text = []
    for line in readme_file_path.read_text().splitlines():
        if not line:
            continue

        if found_section:
            if line.startswith("#"):
                break

            text.append(line)

        elif line.startswith(f"# {section_name}"):
            found_section = True

    return group_elements(text)


def parse_url_data(text: str) -> UrlData:
    match = RGX_URL_MARKDOWN.match(text)
    if not match:
        raise RuntimeError(f"Cannot parse URL data: {repr(text)}")

    title = match.group("title")
    url = match.group("url")
    return UrlData(title=title, url=url)


def parse_element(element_text: list[str]) -> ModelInfo:
    name = element_text[0][1:].strip()
    data: dict = {}
    for text in element_text[1:]:
        text = text.strip().lstrip("-").strip()
        if ":" not in text:
            raise RuntimeError(f"Cannot parse element: {text} in {element_text}")

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


def parse_elements(elements_text: list[list[str]]) -> list[ModelInfo]:
    return [parse_element(element_text) for element_text in elements_text]


def main(readme_file_path: Path, section_name: str, save_file_path: Path) -> None:
    elements_text = read_section_elements(readme_file_path, section_name)
    parsed_elements = parse_elements(elements_text)
    save_file_path.write_text(
        json.dumps(
            [parsed_element.dict() for parsed_element in parsed_elements],
            ensure_ascii=False,
            indent=4,
        )
    )


if __name__ == "__main__":
    main(Path("README.md"), "Model lists", Path("model_data_list.json"))
