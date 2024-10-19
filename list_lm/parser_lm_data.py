import logging
import re

from list_lm.data import ArticleDataExtended, SuggestedModelInfo, UnsupportedUrl, UrlData, UrlType
from list_lm.ollama_client import OllamaClient
from list_lm.parse_html import parse_arxiv
from list_lm.parse_url import parse_url
from list_lm.prompt_template import get_model_name_prompt

REGEX_ARXIV_MODEL_NAME_FROM_TITLE = re.compile(
    r"^(?P<name>\w+):",
    flags=re.IGNORECASE,
)
REGEX_ARXIV_MODEL_NAME_FROM_ABSTRACT = re.compile(
    r"""(?:we present|we introduce) (?P<name>\w+)[,.:]""",
    flags=re.IGNORECASE,
)

LOGGER = logging.getLogger(__name__)


class ParserLMData:
    def __init__(self) -> None:
        self.ollama_client = OllamaClient()
        self.enable_caching = True

    def parse_url(self, url: str, ollama_model_name: str) -> SuggestedModelInfo | UnsupportedUrl:
        url_type = parse_url(url)
        if url_type == UrlType.ARXIV:
            return self.auto_parse_arxiv(url, ollama_model_name)

        else:
            LOGGER.error(f"Unsupported URL: {url}")
            return UnsupportedUrl(url=url)

    def auto_parse_arxiv(self, url: str, ollama_model_name: str) -> SuggestedModelInfo:
        article_data = parse_arxiv(url, caching=self.enable_caching)
        model_names = self.get_model_names_from_arxiv_data(article_data)

        prompt = get_model_name_prompt(article_data.title, article_data.abstract if article_data.abstract else "")
        model_name = self.ollama_client.generate(ollama_model_name, prompt)
        model_names.append(model_name)

        model_names = self.remove_duplicates(model_names)

        return SuggestedModelInfo(suggested_model_names=model_names, article_data=article_data)

    @staticmethod
    def get_model_names_from_arxiv_data(article_data: ArticleDataExtended) -> list[str]:
        model_names = []
        for regex_pattern, data_search in [
            (REGEX_ARXIV_MODEL_NAME_FROM_TITLE, article_data.title),
            (REGEX_ARXIV_MODEL_NAME_FROM_ABSTRACT, article_data.abstract),
        ]:
            if not data_search:
                continue

            match = regex_pattern.search(data_search)
            if match:
                model_names.append(match.group("name"))

        return model_names

    @staticmethod
    def remove_duplicates(names: list[str]) -> list[str]:
        names_set = set()
        names_uniq = []
        for name in names:
            if name not in names_set:
                names_set.add(name)
                names_uniq.append(name)
        return names_uniq

    @staticmethod
    def parse_code_url(url: str) -> UrlData | str:
        url_type = parse_url(url)
        if url_type == UrlType.GITHUB:
            return UrlData(url=url, title="GitHub")

        return "Unsupported code URL!"

    @staticmethod
    def parse_model_weights_url(url: str) -> UrlData:
        url_type = parse_url(url)
        if url_type == UrlType.HUGGINGFACE:
            return UrlData(url=url, title="HuggingFace models")

        return UrlData(url=url, title="Direct link")
