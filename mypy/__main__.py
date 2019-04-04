"""Mypy type checker command line tool."""

import sys
from mypy.main import main
from mypy.util import handle_keyboard_interrupt


def console_entry() -> None:
    with handle_keyboard_interrupt():
        main(None, sys.stdout, sys.stderr)


if __name__ == '__main__':
    console_entry()
