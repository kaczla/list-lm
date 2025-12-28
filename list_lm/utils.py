import re
import time
from datetime import date, datetime
from functools import wraps
from typing import Callable, ParamSpec, Type, TypeVar

from loguru import logger

TYPE_INPUT = ParamSpec("TYPE_INPUT")
TYPE_RETURN = TypeVar("TYPE_RETURN")

RGX_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


def is_valid_date_string(text: str) -> bool:
    return bool(RGX_DATE.match(text))


def convert_date_to_string(date_to_convert: date) -> str:
    return date_to_convert.strftime("%Y-%m-%d")


def convert_string_to_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Type[Exception] | tuple[Type[Exception], ...] = Exception,
) -> Callable[[Callable[TYPE_INPUT, TYPE_RETURN]], Callable[TYPE_INPUT, TYPE_RETURN]]:
    """
    Decorator that retries a function if it raises an exception.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each attempt (default: 2.0)
        exceptions: Exception type(s) to catch and retry on (default: Exception)

    Returns:
        The decorated function's result if successful, or raises the last exception

    Example:
        @retry_on_failure(max_attempts=3, delay=0.5)
        def unstable_function():
            # Some operation that might fail
            pass
    """

    def decorator(func: Callable[TYPE_INPUT, TYPE_RETURN]) -> Callable[TYPE_INPUT, TYPE_RETURN]:
        @wraps(func)
        def wrapper(*args: TYPE_INPUT.args, **kwargs: TYPE_INPUT.kwargs) -> TYPE_RETURN:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    if attempt > 0:
                        logger.debug(f"Retry attempt {attempt + 1}/{max_attempts} after failure")
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.debug(f"Retry ({attempt + 1}/{max_attempts}) failed with {type(e).__name__}: {e}")

                    time.sleep(current_delay)
                    current_delay *= backoff

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

            raise RuntimeError("Max retry attempts exceeded, but no exception was caught")

        return wrapper

    return decorator
