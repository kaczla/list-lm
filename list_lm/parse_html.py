import logging
from datetime import datetime
from pathlib import Path

import requests
from lxml import html

from list_lm.data import ArticleData, CacheArticleData
from list_lm.data_utils import load_base_model, save_base_model

LOGGER = logging.getLogger(__name__)

CACHE_FILE_ARXIV = Path(".cache/cache_arxiv.json")


def get_html_string(url: str) -> str:
    LOGGER.info(f"Getting HTML content from: {url}")
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        msg = f"Cannot get HTML content for: {url!r} (status: {response.status_code}, text: {response.text!r})"
        LOGGER.error(msg)
        raise RuntimeError(msg)

    LOGGER.info("Collected HTML content")
    return response.text


def parse_arxiv(url: str, caching: bool = True) -> ArticleData:
    cache: CacheArticleData | None = None
    if caching and CACHE_FILE_ARXIV.exists():
        cache = load_base_model(CACHE_FILE_ARXIV, CacheArticleData)
        if url in cache.url_to_article_data:
            return cache.url_to_article_data[url]

    html_string = get_html_string(url)
    LOGGER.info("Parsing HTML content")
    html_tree = html.fromstring(html_string.encode())
    title = str(html_tree.xpath("//meta[@name='citation_title']/@content")[0]).strip()
    date_str = str(html_tree.xpath("//meta[@name='citation_date']/@content")[0]).strip()
    converted_date = datetime.strptime(date_str, "%Y/%m/%d").date()
    parsed_data = ArticleData(title=title, url=url, date_create=converted_date)
    LOGGER.info(f"HTML content parsed: {parsed_data}")

    if caching:
        cache = cache if cache is not None else CacheArticleData(url_to_article_data={})
        cache.url_to_article_data[url] = parsed_data
        save_base_model(CACHE_FILE_ARXIV, cache)

    return parsed_data
