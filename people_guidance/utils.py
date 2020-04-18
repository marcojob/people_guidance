import pathlib
import logging
import os

import coloredlogs

ROOT_DIR = pathlib.Path(__file__).parent.parent
ROOT_LOG_DIR = ROOT_DIR / "logs"
ROOT_DATA_DIR = ROOT_DIR / "data"
#DEFAULT_DATASET = ROOT_DATA_DIR / "indoor_dataset_4"
DEFAULT_DATASET = pathlib.Path("/media/marco/SSD/indoor_dataset_5")

def project_path(relative_path: str) -> pathlib.Path:
    return ROOT_DIR / relative_path


def init_logging() -> None:
    logging.basicConfig(level=logging.DEBUG)



def get_logger(name: str, log_dir: pathlib.Path, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logfile = log_dir / f"{name}_{os.getpid()}.log"

    coloredlogs.install(logger=logger, level=level)

    fh = logging.FileHandler(str(logfile))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
