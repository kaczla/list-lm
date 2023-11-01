import logging
from pathlib import Path

from list_lm.data import ApplicationData, get_application_data_sort_key
from list_lm.data_utils import load_base_model_list, save_base_model_list
from list_lm.parse_links import MAP_LINK_TYPE_NAME_TO_FILE_NAME

LOGGER = logging.getLogger(__name__)


def validate_links() -> None:
    changed = False
    errors = []
    normalized_names_to_application_data: dict[str, ApplicationData] = {}
    application_data: ApplicationData
    for file_name in MAP_LINK_TYPE_NAME_TO_FILE_NAME.values():
        path = Path(f"data/json/{file_name}.json")
        application_data_list = load_base_model_list(path, ApplicationData)

        LOGGER.info(f"Processing file name: {file_name}")
        for index, application_data in enumerate(application_data_list):
            LOGGER.info(
                f"[{file_name}][{index + 1}/{len(application_data_list)}] Checking model: {application_data.name}"
            )

            name = application_data.name.lower()
            original_application_data = normalized_names_to_application_data.get(name)
            if original_application_data:
                error_msg = (
                    f"Duplicated link in {file_name} - {application_data.name}"
                    f" (original: {original_application_data})"
                )
                errors.append(error_msg)
            else:
                normalized_names_to_application_data[name] = application_data

            # Here can be a logic for fixing issues in data

        if changed:
            LOGGER.info(f"Saving in: {path}")
            save_base_model_list(path, application_data_list, sort_fn=get_application_data_sort_key)
        else:
            LOGGER.info("Nothing changed - skip saving")

    if errors:
        for error_msg in errors:
            LOGGER.error(error_msg)
        LOGGER.info(f"Found {len(errors)} errors")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    validate_links()


if __name__ == "__main__":
    main()
