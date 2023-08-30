import json
from pathlib import Path

from list_lm.data import ModelInfo


def parse_elements(data_file_path: Path) -> list[ModelInfo]:
    return [ModelInfo(**data) for data in json.loads(data_file_path.read_text())]


def save_markdown(save_path: Path, section_name: str, model_list: list[ModelInfo]) -> None:
    text = f"# {section_name}\n\n"
    text += "\n\n".join([model_data.to_markdown_element() for model_data in model_list])
    save_path.write_text(text)


def main(data_file_path: Path, save_file_path: Path, section_name: str) -> None:
    model_list = parse_elements(data_file_path)
    save_markdown(save_file_path, section_name, model_list)


if __name__ == "__main__":
    main(Path("model_data_list.json"), Path("tmp-README.md"), "Model lists")
