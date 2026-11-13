"""Logging with a console handler and a rotating file handler.

Never log API keys or full prompts containing private user data.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from . import config

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("researchpilot")
    if not _CONFIGURED:
        root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            fh = RotatingFileHandler(
                os.path.join(config.LOG_DIR, "researchpilot.log"),
                maxBytes=1_000_000, backupCount=3, encoding="utf-8",
            )
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            root.warning("File logging unavailable; using console only.")
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("researchpilot") else f"researchpilot.{name}")
