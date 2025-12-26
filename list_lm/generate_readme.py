from pathlib import Path

from loguru import logger

from list_lm.data import ApplicationData, LinkType, ModelInfo
from list_lm.data_utils import load_base_model_list
from list_lm.log_utils import init_logs
from list_lm.parse_links import FILE_NAME_LINKS, MAP_LINK_TYPE_NAME_TO_NORMALISED_NAME
from list_lm.parse_lm_data import FILE_NAME_LM_DATA


def generate_lm_data() -> None:
    model_data_list = load_base_model_list(Path(f"data/json/{FILE_NAME_LM_DATA}.json"), ModelInfo)
    with Path("data/readme/language_models.md").open("wt") as f_write:
        text_to_save = "\n\n".join([model_data.to_markdown_element() for model_data in model_data_list])
        f_write.write(text_to_save)
        f_write.write("\n")
    logger.info("Generated README for LM data")


def generate_links_selected(application_data_list: list[ApplicationData], link_type: LinkType) -> None:
    file_type = MAP_LINK_TYPE_NAME_TO_NORMALISED_NAME[link_type]
    with Path(f"data/readme/{file_type}.md").open("wt") as f_write:
        for application_data in application_data_list:
            if application_data.link_type == link_type:
                f_write.write(f"- {application_data.to_markdown()}\n")
        logger.info(f"Generated README for {file_type}")


def generate_links_all() -> None:
    application_data_list = load_base_model_list(Path(f"data/json/{FILE_NAME_LINKS}.json"), ApplicationData)
    for file_type in [*LinkType]:
        generate_links_selected(application_data_list, file_type)


if __name__ == "__main__":
    init_logs()
    generate_lm_data()
    generate_links_all()
