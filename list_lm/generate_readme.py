import json
from pathlib import Path

from list_lm.data import ApplicationData, ModelInfo


def generate_lm_data() -> None:
    json_data = json.loads(Path("data/json/model_data_list.json").read_text())
    model_list = [ModelInfo(**single_data) for single_data in json_data]
    with Path("data/readme/language_models.md").open("wt") as f_write:
        text_to_save = "\n\n".join([model_data.to_markdown_element() for model_data in model_list])
        f_write.write(text_to_save)
        f_write.write("\n")


def generate_links_selected(file_name: str) -> None:
    json_data = json.loads(Path(f"data/json/{file_name}.json").read_text())
    parsed_data = [ApplicationData(**single_data) for single_data in json_data]
    with Path(f"data/readme/{file_name}.md").open("wt") as f_write:
        for app_data in parsed_data:
            f_write.write(f"- {app_data.to_markdown()}\n")


def generate_links_all() -> None:
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
        generate_links_selected(file_name)


if __name__ == "__main__":
    generate_lm_data()
    generate_links_all()
