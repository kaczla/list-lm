import logging
from pathlib import Path

from list_lm.data import ModelInfo, get_model_info_sort_key
from list_lm.data_utils import load_base_model_list, save_base_model_list
from list_lm.parse_html import parse_arxiv
from list_lm.parse_lm_data import FILE_NAME_LM_DATA

LOGGER = logging.getLogger(__name__)


def validate_lm_data(path: Path, check_model_names: bool) -> None:
    model_info_list = load_base_model_list(path, ModelInfo)
    model_name_to_model_info: dict[str, ModelInfo] = {}
    title_to_model_info: dict[str, ModelInfo] = {}
    url_to_model_info: dict[str, ModelInfo] = {}

    errors = []
    changed = False
    model_info: ModelInfo

    # Check if model name is present in the title or abstract of publication
    if check_model_names:
        for model_info in model_info_list:
            if model_info.manual_validated:
                continue

            if "arxiv" in model_info.publication.url:
                article_data = parse_arxiv(model_info.publication.url)
                model_name = model_info.name

                model_names = []
                if "(" in model_name:
                    search_substring_index = model_name.index("(")
                    model_names.append(model_name[:search_substring_index].strip())
                    model_names.append(model_name[search_substring_index:].strip().strip("()").strip())
                elif "/" in model_name:
                    search_substring_index = model_name.index("/")
                    model_names.append(model_name[:search_substring_index].strip())
                    model_names.append(model_name[search_substring_index:].strip().strip("/").strip())
                elif "-" in model_name:
                    model_names.append(model_name)
                    model_names.append(model_name.replace("-", " "))
                else:
                    model_names.append(model_name)

                for model_name in model_names:
                    model_name = model_name.lower()
                    if model_name in article_data.title.lower() or (
                        article_data.abstract and model_name in article_data.abstract.lower()
                    ):
                        break
                else:
                    errors.append(f"Cannot find model name: {model_info.name!r} in publication title/abstract.")

    for index, model_info in enumerate(model_info_list):
        LOGGER.info(f"[{index + 1}/{len(model_info_list)}] Checking model: {model_info.name}")
        model_name = model_info.name
        if model_name in model_name_to_model_info:
            errors.append(f"Duplicated model name - {model_name}")
        else:
            model_name_to_model_info[model_name] = model_info

        title = model_info.publication.title
        if title in title_to_model_info:
            errors.append(f"Duplicated publication title {title!r} for: {model_name}")
        else:
            title_to_model_info[title] = model_info

        url = model_info.publication.url
        if url in url_to_model_info:
            errors.append(f"Duplicated publication URL {url} for: {model_name}")
        else:
            url_to_model_info[url] = model_info

        # Here can be a logic for fixing issues in data

    if errors:
        for error_msg in errors:
            LOGGER.error(error_msg)
        LOGGER.info(f"Found {len(errors)} errors")

    if changed:
        LOGGER.info(f"Saving in: {path}")
        save_base_model_list(path, model_info_list, sort_fn=get_model_info_sort_key)  # type: ignore[arg-type]
    else:
        LOGGER.info("Nothing changed - skip saving")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    validate_lm_data(Path(f"data/json/{FILE_NAME_LM_DATA}.json"), check_model_names=True)


if __name__ == "__main__":
    main()
