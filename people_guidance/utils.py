import pathlib
import logging
import os

import coloredlogs

ROOT_DIR = pathlib.Path(__file__).parent.parent
ROOT_LOG_DIR = ROOT_DIR / "logs"


def project_path(relative_path: str) -> pathlib.Path:
    return ROOT_DIR / relative_path


def init_logging() -> None:
    logging.basicConfig(level=logging.DEBUG)
    coloredlogs.install(level=logging.DEBUG)


def get_logger(name: str, log_dir: pathlib.Path):
    logger = logging.getLogger(name)
    logfile = log_dir / f"{name}_{os.getpid()}.log"

    fh = logging.FileHandler(str(logfile))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
