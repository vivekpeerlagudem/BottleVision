"""Logging setup.

Responsibility: provide one consistent way to configure and obtain loggers so
every module logs in the same format. Keeps ``print`` out of the codebase.

Usage::

    from bottlevision.utils.logging import configure_logging, get_logger

    configure_logging()          # call once, at program startup
    log = get_logger(__name__)   # call anywhere you need to log
    log.info("hello")
"""

from __future__ import annotations

import logging

# A compact, readable line format: time | LEVEL | module | message
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once for the whole application.

    Args:
        level: The minimum severity to emit (e.g. ``logging.INFO``).
    """
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Usually ``__name__`` of the calling module, which produces a
            logger hierarchy that mirrors the package layout.

    Returns:
        A standard library :class:`logging.Logger`.
    """
    return logging.getLogger(name)
