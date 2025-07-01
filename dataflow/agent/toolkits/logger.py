import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from termcolor import colored
import os


class ColoredFormatter(logging.Formatter):
    # level → (foreground, attrs)
    STYLE = {
        'DEBUG'   : ('cyan',   ['bold']),          # 鲜艳青色
        'INFO'    : ('green',  ['bold']),          # 亮绿色
        'WARNING' : ('yellow', ['bold']),          # 亮黄色
        'ERROR'   : ('red',    ['bold']),          # 亮红色
        'CRITICAL': ('red',    ['bold', 'blink', 'reverse']),  # 红底白字闪烁
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        fg, attrs = self.STYLE.get(record.levelname, ('white', []))
        return colored(msg, fg, attrs=attrs)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Initialize logging configuration.

    Args:
        log_dir (str): The directory where log files will be saved.
        log_level (str): Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    """
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    file_handler = RotatingFileHandler(
        Path(log_dir) / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)