from urllib.parse import urlparse

from loguru import logger

from list_lm.data import UrlType

URL_NETLOC_TO_URL_TYPE = {
    "arxiv.org": UrlType.ARXIV,
    "github.com": UrlType.GITHUB,
    "huggingface.co": UrlType.HUGGINGFACE,
    "twitter.com": UrlType.X,
    "x.com": UrlType.X,
    "dl.acm.org": UrlType.ACM,
}


def parse_url(url: str) -> UrlType:
    parsed_url = urlparse(url)
    base_url = parsed_url.netloc
    base_url = base_url.removeprefix("www.")
    url_type = URL_NETLOC_TO_URL_TYPE.get(base_url, UrlType.UNKNOWN)
    logger.debug(f"Parsed URL {url!r} ({base_url!r}) as {url_type}")
    return url_type
