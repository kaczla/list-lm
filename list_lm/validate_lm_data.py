import logging
from pathlib import Path

from list_lm.data import ModelInfo, get_model_info_sort_key
from list_lm.data_utils import load_base_model_list, save_base_model_list

LOGGER = logging.getLogger(__name__)


def validate_lm_data(path: Path) -> None:
    model_info_list = load_base_model_list(path, ModelInfo)
    model_name_to_model_info: dict[str, ModelInfo] = {}

    changed = False
    errors = []
    model_info: ModelInfo
    for index, model_info in enumerate(model_info_list):
        LOGGER.info(f"[{index + 1}/{len(model_info_list)}] Checking model: {model_info.name}")
        if model_info.name in model_name_to_model_info:
            error_msg = f"Duplicated model name - {model_info.name}"
            errors.append(error_msg)
        else:
            model_name_to_model_info[model_info.name] = model_info
        # Here can be a logic for fixing issues in data

    if errors:
        for error_msg in errors:
            LOGGER.error(error_msg)
        LOGGER.info(f"Found {len(errors)} errors")

    if changed:
        LOGGER.info(f"Saving in: {path}")
        save_base_model_list(path, model_info_list, sort_fn=get_model_info_sort_key)
    else:
        LOGGER.info("Nothing changed - skip saving")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    validate_lm_data(Path("data/json/model_data_list.json"))


if __name__ == "__main__":
    main()
