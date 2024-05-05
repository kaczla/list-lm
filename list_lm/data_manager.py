from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from list_lm.data_utils import SORT_FN_TYPING, load_base_model_list, save_base_model_list

T = TypeVar("T", bound=BaseModel)


class DataManager(Generic[T]):
    data: list[T]

    def __init__(self, data_path: Path, object_type: type[T], sort_fn: SORT_FN_TYPING | None) -> None:
        self.data_path = data_path
        self.object_type = object_type
        self.sort_fn = sort_fn
        self.data = []
        self.load_data()

    def load_data(self) -> None:
        if self.data_path.exists():
            self.data = load_base_model_list(self.data_path, self.object_type)

    def save(self) -> None:
        if self.sort_fn:
            self.data = sorted(self.data, key=self.sort_fn)

        save_base_model_list(self.data_path, self.data)

    def add(self, element: T) -> None:
        self.data.append(element)
        self.save()

    def get_data(self) -> list[T]:
        return self.data
