import re
from datetime import date, datetime

RGX_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


def is_valid_date_string(text: str) -> bool:
    return bool(RGX_DATE.match(text))


def convert_date_to_string(date_to_convert: date) -> str:
    return date_to_convert.strftime("%Y-%m-%d")


def convert_string_to_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()
