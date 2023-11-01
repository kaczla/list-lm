import logging
from pathlib import Path

from list_lm.data import ApplicationData, get_application_data_sort_key
from list_lm.data_utils import load_base_model_list, save_base_model_list
from list_lm.parse_links import FILE_NAME_LINKS

LOGGER = logging.getLogger(__name__)


def validate_links() -> None:
    changed = False
    errors = []
    warnings = []
    normalized_names_to_application_data_list: dict[str, list[ApplicationData]] = {}
    path = Path(f"data/json/{FILE_NAME_LINKS}.json")
    application_data_list = load_base_model_list(path, ApplicationData)

    application_data: ApplicationData
    for index, application_data in enumerate(application_data_list):
        LOGGER.info(
            f"[{index + 1}/{len(application_data_list)}]"
            f" Checking model: {application_data.name} ({application_data.type_name})"
        )

        normalized_name = application_data.name.lower()
        original_application_data_list = normalized_names_to_application_data_list.get(normalized_name)
        if original_application_data_list:
            for original_application_data in original_application_data_list:
                if original_application_data.name.lower() == normalized_name:
                    if original_application_data.url == application_data.url:
                        error_msg = (
                            f"Duplicated link in {application_data.type_name} - {application_data.name},"
                            f" found original: {original_application_data})"
                        )
                        errors.append(error_msg)
                        break

                    warning_msg = (
                        f"Potential duplicate, duplicated name in {application_data.type_name}"
                        f" - {application_data.name},"
                        f" found original: {original_application_data})"
                    )
                    warnings.append(warning_msg)

            else:
                normalized_names_to_application_data_list[normalized_name].append(application_data)

        else:
            normalized_names_to_application_data_list[normalized_name] = [application_data]

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

    if warnings:
        for warning_msg in warnings:
            LOGGER.warning(warning_msg)
        LOGGER.info(f"Found {len(warnings)} warnings")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    validate_links()


if __name__ == "__main__":
    main()
