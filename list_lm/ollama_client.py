import logging
from urllib.parse import urljoin

import requests
from requests import RequestException

LOGGER = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, url: str = "http://localhost:11434") -> None:
        self.url = url
        self.url_list_models = urljoin(self.url, "api/tags")
        self.url_generate = urljoin(self.url, "api/generate")

    def is_available(self) -> bool:
        LOGGER.debug("Checking if Ollama is available...")
        try:
            response = requests.get(self.url, timeout=60)
        except RequestException as error:
            LOGGER.warning(f"Cannot connect to Ollama service, error: {error}")
            return False
        is_available = response.status_code == 200
        LOGGER.info(f"Ollama is available: {is_available}, response: {response.text!r}")
        return is_available

    def get_model_list(self) -> list[str]:
        response = requests.get(self.url_list_models, timeout=60)
        response_json = response.json()
        try:
            return [str(model_data["name"]) for model_data in response_json["models"]]
        except KeyError as error:
            LOGGER.error(f"Cannot get model names, error: {error}")
        return []

    def generate(self, model_name: str, prompt: str) -> str:
        request_input = {"model": model_name, "stream": False, "prompt": prompt}
        LOGGER.debug(f"Generating text for request: {request_input}")
        response = requests.post(self.url_generate, timeout=60, json=request_input)
        response_json = response.json()
        LOGGER.debug(f"Generated response: {response.text!r}")
        return str(response_json.get("response", ""))
