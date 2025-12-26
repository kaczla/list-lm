import re
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger
from lxml import html
from urllib3 import Retry

from list_lm.data import ArticleDataExtended, CacheArticleData, UnparsedUrl
from list_lm.data_utils import load_base_model, save_base_model

CACHE_DIR = Path(".cache")
CACHE_FILE_ARXIV = CACHE_DIR / "cache_arxiv.json"

REGEX_GITHUB_URL = re.compile(r"github.com/(?P<author>[^/]+)/(?P<project>[^/?]+)")


def get_request_session() -> requests.Session:
    adapter = requests.adapters.HTTPAdapter(
        max_retries=Retry(total=5, backoff_factor=0.1, status_forcelist=[413, 429, 500, 502, 503, 504])
    )
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_html_string(url: str) -> str:
    logger.info(f"Getting HTML content from: {url}")
    with get_request_session() as session:
        response = session.get(url, timeout=60)
        if response.status_code != 200:
            msg = f"Cannot get HTML content for: {url!r} (status: {response.status_code}, text: {response.text!r})"
            logger.error(msg)
            raise RuntimeError(msg)

    logger.info("Collected HTML content")
    return response.text


def parse_arxiv(url: str, caching: bool = True) -> ArticleDataExtended:
    cache: CacheArticleData | None = None
    if caching and CACHE_FILE_ARXIV.exists():
        cache = load_base_model(CACHE_FILE_ARXIV, CacheArticleData)
        if url in cache.url_to_article_data:
            return cache.url_to_article_data[url]

    html_string = get_html_string(url)
    logger.info("Parsing HTML content")
    html_tree = html.fromstring(html_string.encode())
    title = str(html_tree.xpath("//meta[@name='citation_title']/@content")[0]).strip()
    date_str = str(html_tree.xpath("//meta[@name='citation_date']/@content")[0]).strip()
    converted_date = datetime.strptime(date_str, "%Y/%m/%d").date()
    abstract_data = html_tree.xpath("//meta[@name='citation_abstract']/@content")
    if abstract_data:
        abstract_str = str(abstract_data[0]).strip()
    else:
        abstract_str = ""
        logger.error(f"Cannot get abstract for: {url}")
    abstract_urls = list(map(str, html_tree.xpath("//div[@id='abs']/blockquote[contains(@class,'abstract')]/a/@href")))
    abstract_urls = sorted(set(abstract_urls))
    parsed_data = ArticleDataExtended(
        title=title,
        url=url,
        date_create=converted_date,
        abstract=abstract_str,
        article_urls=abstract_urls if abstract_urls else None,
    )
    logger.info(f"HTML content parsed: {parsed_data}")

    if caching:
        cache = cache if cache is not None else CacheArticleData(url_to_article_data={})
        cache.url_to_article_data[url] = parsed_data
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir()
        save_base_model(CACHE_FILE_ARXIV, cache)

    return parsed_data


def get_github_readme(url: str) -> str | UnparsedUrl:
    regex_match = REGEX_GITHUB_URL.search(url)
    if not regex_match:
        return UnparsedUrl(url=url, message="Cannot match GitHub URL")

    url_author_name = regex_match.group("author")
    url_project_name = regex_match.group("project")
    for branch_name in ["master", "main"]:
        url = f"https://raw.githubusercontent.com/{url_author_name}/{url_project_name}/{branch_name}/README.md"
        try:
            return get_html_string(url)
        except RuntimeError:
            continue

    return UnparsedUrl(url=url, message="Cannot find README.md in GitHub URL")
