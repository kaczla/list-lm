import logging
from datetime import datetime

import requests
from lxml import html

from list_lm.data import ArticleData

LOGGER = logging.getLogger(__name__)


def get_html_string(url: str) -> str:
    LOGGER.info(f"Getting HTML content from: {url}")
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        msg = f"Cannot get HTML content for: {url!r} (status: {response.status_code}, text: {response.text!r})"
        LOGGER.error(msg)
        raise RuntimeError(msg)

    LOGGER.info("Collected HTML content")
    return response.text


def parse_arxiv(url: str) -> ArticleData:
    html_string = get_html_string(url)
    LOGGER.info("Parsing HTML content")
    html_tree = html.fromstring(html_string.encode())
    title = str(html_tree.xpath("//meta[@name='citation_title']/@content")[0]).strip()
    date_str = str(html_tree.xpath("//meta[@name='citation_date']/@content")[0]).strip()
    converted_date = datetime.strptime(date_str, "%Y/%m/%d").date()
    parsed_data = ArticleData(title=title, date_create=converted_date)
    LOGGER.info(f"HTML content parsed: {parsed_data}")
    return parsed_data
