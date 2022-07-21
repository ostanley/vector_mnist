import logging
from typing import Optional

import torch.nn as nn


def setup_logging(
    log_path: Optional[str] = None,
    log_level: str = "DEBUG",
    print_level: str = "INFO",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Sets up the logger

    Args:
        log_path (str, optional): The path to the log file. Defaults to None.
        log_level (str, optional): Min severity level. Defaults to "DEBUG".
        print_level (str, optional): The default print level. Defaults to
        "INFO".
        logger (logger, optional): An instantiated logger. Defaults to None.
    """
    logger = logger if logger else logging.getLogger()
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    stream_handler.setLevel(print_level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(print_level)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
        logger.info("Log file is %s", log_path)


def init_weights(m: nn.modules) -> None:
    """Initialize weights with Xavier normal initializer

    Args:
        m: Network module
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight)
        nn.init.zeros_(m.bias)
