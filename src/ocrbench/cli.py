"""Command-line interface for ocrbench."""

import argparse

from ocrbench import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the initial command-line parser."""
    parser = argparse.ArgumentParser(prog="ocrbench")
    parser.add_argument(
        "--version", action="store_true", help="show the installed version and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    arguments = build_parser().parse_args(argv)
    if arguments.version:
        print(__version__)
    return 0
