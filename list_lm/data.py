from datetime import date
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel


class ApplicationData(BaseModel):
    name: str
    description: str
    url: str
    type_name: str

    def to_markdown(self) -> str:
        return f"[{self.name}]({self.url}) - {self.description}"


class UrlData(BaseModel):
    title: str
    url: str

    def to_markdown(self) -> str:
        return f"[{self.title}]({self.url})"


class ArticleData(UrlData):
    date_create: date


class ModelInfo(BaseModel):
    name: str
    year: int
    publication: ArticleData
    video: UrlData | None = None
    code: UrlData | None = None
    weights: UrlData | None = None

    def to_markdown_element(self) -> str:
        return "\n".join(
            [
                f"- {self.name}",
                f"  - Year: {self.year}",
            ]
            + self.get_optional_elements_markdown()
        )

    def get_optional_elements_markdown(self) -> list[str]:
        return [
            f"  - {label}: {element.to_markdown()}"
            for label, element in [
                ("Publication", self.publication),
                ("Video", self.video),
                ("Code", self.code),
                ("Model weights", self.weights),
            ]
            if element is not None
        ]


class ModelInfoDict(TypedDict):
    name: str
    year: int
    publication: ArticleData
    video: UrlData | None
    code: UrlData | None
    weights: UrlData | None


class LinkType(StrEnum):
    MODEL = "Model links"
    UTILS = "Utils links"
    GPU_PROFILING = "GPU profiling links"
    VISUALIZATION = "Visualization links"
    VOCABULARY = "Vocabulary links"
    OPTIMIZER = "Optimizer links"
    DATASET = "Dataset links"
    DOCUMENTATION = "Documentation links"

    @staticmethod
    def create_from_value(value: str) -> "LinkType":
        value_clean = value.lower().strip()
        enum: LinkType
        for enum in [*LinkType]:
            if enum.value.lower() == value_clean:
                return enum

        raise ValueError(f"Cannot create enum LinkType for value: {repr(value)}")


def get_model_info_sort_key(data: ModelInfo) -> date:
    return data.publication.date_create


def get_application_data_sort_key(data: ApplicationData) -> str:
    return data.name.lower()
