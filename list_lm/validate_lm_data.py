from pathlib import Path

from loguru import logger

from list_lm.data import ModelInfo, get_model_info_sort_key
from list_lm.data_utils import load_base_model_list, save_base_model_list
from list_lm.generate_readme import generate_lm_data
from list_lm.log_utils import init_logs
from list_lm.parse_html import parse_arxiv
from list_lm.parse_lm_data import FILE_NAME_LM_DATA


def validate_lm_data(
    path: Path,
    check_model_names: bool = False,
    update_publication_data: bool = False,
) -> None:
    model_info_list = load_base_model_list(path, ModelInfo)
    model_name_to_model_info: dict[str, ModelInfo] = {}
    title_to_model_info: dict[str, ModelInfo] = {}
    url_to_model_info: dict[str, ModelInfo] = {}

    errors = []
    changed = False
    model_info: ModelInfo

    if update_publication_data:
        for model_info in model_info_list:
            is_updated = update_publication_data_in_model_info(model_info)
            if is_updated:
                model_info.manual_validated = False
                changed = True

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
        logger.info(f"[{index + 1}/{len(model_info_list)}] Checking model: {model_info.name}")
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

    # Check sort order in data
    sorted_model_info_list = sorted(model_info_list, key=get_model_info_sort_key)
    if model_info_list != sorted_model_info_list:
        changed = True

    if errors:
        for error_msg in errors:
            logger.error(error_msg)
        logger.info(f"Found {len(errors)} errors")

    if changed:
        logger.info(f"Saving in: {path}")
        save_base_model_list(path, model_info_list, sort_fn=get_model_info_sort_key)  # type: ignore[arg-type]
        generate_lm_data()
    else:
        logger.info("Nothing changed - skip saving")


def update_publication_data_in_model_info(model_info: ModelInfo) -> bool:
    is_updated = False

    if "arxiv" in model_info.publication.url:
        article_data = parse_arxiv(model_info.publication.url)

        if article_data.title != model_info.publication.title:
            model_info.publication.title = article_data.title
            is_updated = True

        if article_data.date_create != model_info.publication.date_create:
            model_info.publication.date_create = article_data.date_create
            is_updated = True

    return is_updated


def main() -> None:
    init_logs()
    validate_lm_data(
        Path(f"data/json/{FILE_NAME_LM_DATA}.json"),
        check_model_names=True,
        update_publication_data=True,
    )


if __name__ == "__main__":
    main()
