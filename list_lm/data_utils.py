import json
from datetime import date
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from pydantic import BaseModel

BASE_MODEL_TYPE = TypeVar("BASE_MODEL_TYPE", bound=BaseModel)
SORT_VALUE_TYPE = str | int | float | bool | date

SORT_FN_TYPING = Callable[[BASE_MODEL_TYPE], SORT_VALUE_TYPE] | Callable[[BASE_MODEL_TYPE], tuple[SORT_VALUE_TYPE]]


def load_base_model(file_path: Path, base_model_type: type[BASE_MODEL_TYPE]) -> BASE_MODEL_TYPE:
    return base_model_type(**json.loads(file_path.read_text()))


def load_base_model_list(file_path: Path, base_model_type: type[BASE_MODEL_TYPE]) -> list[BASE_MODEL_TYPE]:
    return [base_model_type(**data) for data in json.loads(file_path.read_text())]


def save_base_model(file_path: Path, data: BASE_MODEL_TYPE) -> None:
    file_path.write_text(json.dumps(data.model_dump(mode="json"), indent=4, ensure_ascii=False))


def save_base_model_list(
    file_path: Path,
    data_list: Sequence[BASE_MODEL_TYPE],
    sort_fn: SORT_FN_TYPING | None = None,
) -> None:
    if sort_fn:
        data_list = sorted(data_list, key=sort_fn)

    file_path.write_text(
        json.dumps(
            [data.model_dump(mode="json") for data in data_list],
            indent=4,
            ensure_ascii=False,
        )
    )
