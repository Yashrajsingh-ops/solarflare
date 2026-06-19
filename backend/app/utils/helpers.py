"""
helpers.py — Miscellaneous utility functions.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Set up a consistent logging format for the whole application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))
