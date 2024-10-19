from pathlib import Path

from list_lm.log_utils import init_logs
from list_lm.parse_lm_data import FILE_NAME_LM_DATA
from list_lm.validate_links import validate_links
from list_lm.validate_lm_data import validate_lm_data


def main() -> None:
    init_logs(warning=True)
    validate_links()
    validate_lm_data(
        Path(f"data/json/{FILE_NAME_LM_DATA}.json"),
        check_model_names=True,
        update_publication_data=True,
    )


if __name__ == "__main__":
    main()
