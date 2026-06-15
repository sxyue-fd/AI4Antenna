# -*- coding: utf-8 -*-

import logging
import os


def create_logger(name: str, log_dir: str, log_filename: str):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # file handler
    fh = logging.FileHandler(os.path.join(log_dir, log_filename))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger