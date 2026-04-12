# utils/logger.py
#
# Centralised logging configuration for ReguGrounded.
# All modules call setup_logger(__name__) to get a pre-configured Logger.
# Two decorators are provided for structured observability:
#   @log_errors    — captures full traceback + execution time on exceptions
#   @log_function_call — logs function entry/exit with truncated arg preview

import functools
import logging
import os
import time
import traceback
from logging.handlers import RotatingFileHandler
from typing import Callable, Optional

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Return a Logger with:
      - StreamHandler  at console_level (INFO by default)
      - RotatingFileHandler at file_level  (DEBUG by default, 10 MB × 5 backups)

    Idempotent: calling setup_logger with the same name twice returns the
    already-configured logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(min(console_level, file_level))

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    log_directory = log_dir or _LOG_DIR
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, f"{name.replace('.', '_')}.log")

    fh = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_errors(logger: logging.Logger) -> Callable:
    """
    Decorator factory — wraps a function so that any exception is logged with
    full traceback and elapsed time before being re-raised.

    Usage:
        @log_errors(logger)
        def my_function(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error(
                    f"{func.__qualname__} raised {type(exc).__name__} "
                    f"after {elapsed:.1f} ms: {exc}\n"
                    + traceback.format_exc()
                )
                raise
        return wrapper
    return decorator


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG) -> Callable:
    """
    Decorator factory — logs function entry (with a truncated preview of string
    args) and exit (with elapsed time).

    Usage:
        @log_function_call(logger)
        def my_function(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build a short preview of string arguments
            previews = []
            for a in args[1:]:  # skip self
                if isinstance(a, str):
                    previews.append(repr(a[:100]) + ("..." if len(a) > 100 else ""))
            for k, v in kwargs.items():
                if isinstance(v, str):
                    previews.append(f"{k}={repr(v[:100])}" + ("..." if len(v) > 100 else ""))

            arg_str = ", ".join(previews) if previews else ""
            logger.log(level, f"→ {func.__qualname__}({arg_str})")

            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.log(level, f"← {func.__qualname__} completed in {elapsed:.1f} ms")
            return result
        return wrapper
    return decorator
