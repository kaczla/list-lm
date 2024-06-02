from datetime import date
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel

from list_lm.utils import convert_date_to_string


class LinkType(StrEnum):
    MODEL = "Model links"
    UTILS = "Utils links"
    GPU_PROFILING = "GPU profiling links"
    VISUALIZATION = "Visualization links"
    VOCABULARY = "Vocabulary links"
    OPTIMIZER = "Optimizer links"
    DATASET = "Dataset links"
    TOPIC = "Topic links"
    DOCUMENTATION = "Documentation links"

    @staticmethod
    def create_from_value(value: str) -> "LinkType":
        value_clean = value.lower().strip()
        enum: LinkType
        for enum in [*LinkType]:
            if enum.value.lower() == value_clean:
                return enum

        raise ValueError(f"Cannot create enum LinkType for value: {repr(value)}")


class ApplicationData(BaseModel):
    name: str
    description: str
    url: str
    link_type: LinkType
    manual_validated: bool = False

    def to_markdown(self) -> str:
        return f"[{self.name}]({self.url}) - {self.description}"


class UrlData(BaseModel):
    title: str
    url: str

    def to_markdown(self) -> str:
        return f"[{self.title}]({self.url})"


class ArticleData(UrlData):
    date_create: date

    def to_markdown(self) -> str:
        return super().to_markdown() + f" ({convert_date_to_string(self.date_create)})"


class ArticleDataExtended(ArticleData):
    abstract: str | None = None
    article_urls: list[str] | None = None

    def to_article_data(self) -> ArticleData:
        return ArticleData(
            title=self.title,
            url=self.url,
            date_create=self.date_create,
        )


class ModelInfo(BaseModel):
    name: str
    year: int
    publication: ArticleData
    video: UrlData | None = None
    code: UrlData | None = None
    weights: UrlData | None = None
    manual_validated: bool = False

    def to_markdown_element(self) -> str:
        return "\n".join(
            [
                f"- {self.name}",
                f"  - Year: {self.year}",
            ]
            + self.get_additional_elements_markdown()
        )

    def get_additional_elements_markdown(self) -> list[str]:
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


class SuggestedModelInfo(BaseModel):
    suggested_model_names: list[str]
    article_data: ArticleDataExtended


class CacheArticleData(BaseModel):
    url_to_article_data: dict[str, ArticleDataExtended]


def get_model_info_sort_key(data: ModelInfo) -> tuple[date, str]:
    return data.publication.date_create, data.name.lower()


def get_application_data_sort_key(data: ApplicationData) -> str:
    return data.name.lower()
