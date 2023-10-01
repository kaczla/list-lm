import json
import logging
from pathlib import Path

from list_lm.data import ModelInfo

LOGGER = logging.getLogger(__name__)


def validate_lm_data(path: Path) -> None:
    model_info_list = [ModelInfo(**data) for data in json.loads(path.read_text())]
    model_name_to_model_info = {}

    changed = False
    errors = []
    for index, model_info in enumerate(model_info_list):
        LOGGER.info(f"[{index + 1}/{len(model_info_list)}] Checking model: {model_info.name}")
        if model_info.name in model_name_to_model_info:
            error_msg = f"Duplicated model name - {model_info.name}"
            errors.append(error_msg)
        else:
            model_name_to_model_info[model_info.name] = model_info

    if errors:
        for error_msg in errors:
            LOGGER.error(error_msg)
        LOGGER.info(f"Found {len(errors)} errors")

    if changed:
        LOGGER.info(f"Saving in: {path}")
        path.write_text(json.dumps([m.model_dump(mode="json") for m in model_info_list], indent=4, ensure_ascii=False))
    else:
        LOGGER.info("Nothing changed - skip saving")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    validate_lm_data(Path("data/json/model_data_list.json"))


if __name__ == "__main__":
    main()
