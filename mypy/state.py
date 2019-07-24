from contextlib import contextmanager
from typing import Optional, Tuple, Iterator

# These are global mutable state. Don't add anything here unless there's a very
# good reason.

# Value varies by file being processed
MYPY = False
if MYPY:
    strict_optional = False
find_occurrences = None  # type: Optional[Tuple[str, str]]


@contextmanager
def strict_optional_set(value: bool) -> Iterator[None]:
    global strict_optional
    saved = globals().get('strict_optional')
    strict_optional = value
    try:
        yield
    finally:
        if saved is not None:
            strict_optional = saved
        else:
            del strict_optional
