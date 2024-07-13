import logging


def init_logs(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if debug:
        logging.getLogger("urllib3").setLevel(logging.INFO)
