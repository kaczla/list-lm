from pathlib import Path

from loguru import logger

from list_lm.log_utils import init_logs
from list_lm.parse_lm_data import FILE_NAME_LM_DATA
from list_lm.validate_links import validate_links
from list_lm.validate_lm_data import validate_lm_data


def main() -> None:
    init_logs(warning=True)

    # Run validations and collect errors/warnings
    link_errors, link_warnings = validate_links()
    lm_data_errors = validate_lm_data(
        Path(f"data/json/{FILE_NAME_LM_DATA}.json"),
        check_model_names=True,
        update_publication_data=True,
    )

    # Display consolidated summary at the end
    total_errors = len(link_errors) + len(lm_data_errors)
    total_warnings = len(link_warnings)

    logger.info("Validation summary:")

    if link_errors:
        logger.info(f"Link validation errors: {len(link_errors)}")
        for error in link_errors:
            logger.error(f"  - {error}")

    if lm_data_errors:
        logger.info(f"LM data validation errors: {len(lm_data_errors)}")
        for error in lm_data_errors:
            logger.error(f"  - {error}")

    if link_warnings:
        logger.info(f"Link validation warnings: {len(link_warnings)}")
        for warning in link_warnings:
            logger.warning(f"  - {warning}")

    logger.info(f"Total: {total_errors} error(s), {total_warnings} warning(s)")


if __name__ == "__main__":
    main()
