from urllib.parse import urlparse

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
    return URL_NETLOC_TO_URL_TYPE.get(parsed_url.netloc, UrlType.UNKNOWN)
