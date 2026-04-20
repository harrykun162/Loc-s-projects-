"""
src/utils/logger.py
-------------------
Centralised structured logging. Every module calls get_logger(__name__).
Writes to console (INFO) and rotating file (DEBUG).
"""
from __future__ import annotations
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_path = log_file or os.getenv("LOG_FILE", "artifacts/app.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
