#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)


@dataclass
class Section:
    name: str
    text: List[str]

    def text_to_line(self) -> str:
        return "\n".join(self.text)

    def to_text(self, extra_new_line: bool = False) -> str:
        text = f"# {self.name}\n\n{self.text_to_line()}\n"
        if extra_new_line:
            text += "\n"
        return text


def parse_section(text: List[str]) -> Section:
    section_name = text.pop(0)[1:].strip()

    # Find empty lines at the beginning
    indexes_to_remove = []
    for index, text_line in enumerate(text):
        if not text_line.strip():
            indexes_to_remove.append(index)
            continue

        break

    # Return empty text if it is
    if len(indexes_to_remove) == len(text):
        LOGGER.error(f"Empty text in section: {section_name}")
        return Section(name=section_name, text=[])

    # Find empty lines at the end
    for index, text_line in enumerate(reversed(text)):
        if not text_line.strip():
            indexes_to_remove.append(len(text) - 1 - index)
            continue

        break

    # Remove empty lines
    for index in sorted(indexes_to_remove, reverse=True):
        del text[index]

    return Section(name=section_name, text=text)


def read_readme(file_path: Path) -> List[Section]:
    sections, text_lines = [], []

    for line in file_path.read_text().split("\n"):
        if line.startswith("# ") and text_lines:
            sections.append(parse_section(text_lines))
            text_lines = []

        text_lines.append(line.rstrip())

    if text_lines:
        sections.append(parse_section(text_lines))

    LOGGER.info(f"Found {len(sections)} sections")
    LOGGER.info(f"Sections: {[s.name for s in sections]}")
    return sections


def sort_links(text: List[str]) -> List[str]:
    link_with_names = []
    for i_link_text in text:
        start_bracket_idx = i_link_text.find("[")
        end_bracket_idx = i_link_text.find("]")
        if start_bracket_idx < 0 or end_bracket_idx < 0:
            link_with_names.append(("", i_link_text))
        else:
            link_with_names.append((i_link_text[start_bracket_idx + 1 : end_bracket_idx].lower(), i_link_text,))
    link_with_names.sort(key=lambda x: x[0])
    return [text for _, text in link_with_names]


def clean_markdown_link(link_text: str) -> str:
    start_name_index, end_name_index = link_text.find("["), link_text.find("]")
    start_url_index, end_url_index = link_text.rfind("("), link_text.rfind(")")
    if start_name_index < 0 or end_name_index < 0 or start_url_index < 0 or end_url_index < 0:
        LOGGER.warning(f"Cannot detect name and URL from markdown link for: {repr(link_text)}")
        return link_text

    link_prefix = link_text[:start_name_index] if start_name_index > 0 else ""
    link_suffix = link_text[end_url_index + 1 :] if (end_url_index + 1) < len(link_text) else ""
    link_name = link_text[start_name_index + 1 : end_name_index].strip()
    link_url = link_text[start_url_index + 1 : end_url_index].strip()
    if link_url.endswith("/"):
        link_url = link_url[:-1]

    return f"{link_prefix}[{link_name}]({link_url}){link_suffix}"


def clean_text(text: List[str], is_link: bool) -> List[str]:
    cleaned_text = []
    for i_text in text:
        i_text = i_text.strip()
        if is_link:
            separator = " - "
            count_separator = i_text.count(separator)
            if count_separator != 1:
                LOGGER.warning(f"Found {count_separator} link-text separator - expected 1 in {repr(i_text)}")
                cleaned_text.append(i_text)
                continue

            index = i_text.find(separator)
            prefix_with_link = i_text[:index].strip()
            prefix_with_link = clean_markdown_link(prefix_with_link)
            description = i_text[index + 3 :].strip()
            i_text = f"{prefix_with_link}{separator}{description}"
        cleaned_text.append(i_text)
    return cleaned_text


def process_sections_links(sections: List[Section], sort: bool = True) -> List[Section]:
    processed_sections = []

    for section in sections:
        if not section.name.endswith("links"):
            processed_sections.append(section)
            continue

        section.text = clean_text(section.text, True)
        if sort:
            section.text = sort_links(section.text)
        processed_sections.append(section)

    return processed_sections


def write_readme(sections: List[Section], save_path: Path) -> None:
    LOGGER.info(f"Writing into: {save_path}")
    text_to_save = "\n".join([section.to_text() for section in sections])
    save_path.write_text(text_to_save)


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md does not exist!"

    sections = process_sections_links(read_readme(readme_path))
    write_readme(sections, readme_path)


if __name__ == "__main__":
    main()
