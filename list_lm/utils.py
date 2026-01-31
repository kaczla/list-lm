import re
from datetime import date, datetime

RGX_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
RGX_NAME_WITH_PARENS = re.compile(r"^(.+?)\s*\((.+)\)$")


def is_valid_date_string(text: str) -> bool:
    return bool(RGX_DATE.match(text))


def convert_date_to_string(date_to_convert: date) -> str:
    return date_to_convert.strftime("%Y-%m-%d")


def convert_string_to_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def normalize_name_format(name: str) -> str:
    """
    Normalize name format from 'SHORTCUT (Full Name)' to 'Full Name (SHORTCUT)'.

    Detects if the part before parentheses is an acronym (short, uppercase)
    and the part inside is a full name (longer, contains spaces), then swaps them.
    """
    match = RGX_NAME_WITH_PARENS.match(name)
    if not match:
        return name

    before_parens = match.group(1).strip()
    inside_parens = match.group(2).strip()

    # Check if before_parens looks like an acronym:
    # - Short (≤10 chars)
    # - No spaces
    # - Mostly uppercase letters (allow digits and hyphens)
    is_before_acronym = (
        len(before_parens) <= 10
        and " " not in before_parens
        and sum(1 for c in before_parens if c.isupper()) > len(before_parens) // 2
    )

    # Check if inside_parens looks like a full name:
    # - Contains spaces (multiple words)
    # - Longer than before_parens
    is_inside_full_name = " " in inside_parens and len(inside_parens) > len(before_parens)

    if is_before_acronym and is_inside_full_name:
        return f"{inside_parens} ({before_parens})"

    return name
