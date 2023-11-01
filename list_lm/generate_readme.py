from pathlib import Path

from list_lm.data import ApplicationData, ModelInfo
from list_lm.data_utils import load_base_model_list


def generate_lm_data() -> None:
    model_data_list = load_base_model_list(Path("data/json/model_data_list.json"), ModelInfo)
    with Path("data/readme/language_models.md").open("wt") as f_write:
        text_to_save = "\n\n".join([model_data.to_markdown_element() for model_data in model_data_list])
        f_write.write(text_to_save)
        f_write.write("\n")


def generate_links_selected(file_name: str) -> None:
    application_data_list = load_base_model_list(Path(f"data/json/{file_name}.json"), ApplicationData)
    with Path(f"data/readme/{file_name}.md").open("wt") as f_write:
        for application_data in application_data_list:
            f_write.write(f"- {application_data.to_markdown()}\n")


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
