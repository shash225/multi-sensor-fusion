"""Logging setup shared by training, evaluation, scripts and the dashboard."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO, filename: str = "run.log") -> None:
    """Configure the root logger once (console) and add a file handler per ``log_dir``."""
    global _configured
    root = logging.getLogger()
    if not _configured:
        root.setLevel(level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(console)
        _configured = True
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        target = str((log_dir / filename).resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target for h in root.handlers):
            handler = logging.FileHandler(target, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class WarnOnce:
    """Emit a warning only the first time a key is seen (e.g. once per corrupt file)."""

    def __init__(self, logger: logging.Logger, limit: int = 50) -> None:
        self.logger = logger
        self.seen: set[str] = set()
        self.limit = limit

    def __call__(self, key: str, message: str) -> None:
        if key in self.seen or len(self.seen) >= self.limit:
            return
        self.seen.add(key)
        self.logger.warning(message)
