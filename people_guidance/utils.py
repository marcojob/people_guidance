import pathlib
import logging
import os
import numpy as np

import coloredlogs

ROOT_DIR = pathlib.Path(__file__).parent.parent
ROOT_LOG_DIR = ROOT_DIR / "logs"
ROOT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATASET = ROOT_DATA_DIR / "indoor_dataset_6"

INTRINSIC_MATRIX = np.array([[1.29168322e+03, 0.0, 8.10433936e+02], [0.0, 1.29299333e+03, 6.15008893e+02], [0.0, 0.0, 1.0]])

DISTORTION_COEFFS = np.array([[ 0.1952957 , -0.48124548, -0.00223218, -0.00106617,  0.2668875 ]])

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
