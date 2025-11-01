from __future__ import annotations
import logging
from dataclasses import dataclass

@dataclass
class LogConfig:
    level: int = logging.INFO
    fmt: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt: str = "%H:%M:%S"

def configure_logging(level: int | None = None) -> None:
    logging.basicConfig(level=level or LogConfig.level,
                        format=LogConfig.fmt, datefmt=LogConfig.datefmt)
