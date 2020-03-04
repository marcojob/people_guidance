import multiprocessing as mp
import pathlib
import logging
import traceback
from typing import Optional, Any, Dict, List, Tuple

from ..utils import get_logger


class Module:
    def __init__(self, name: str, log_dir: pathlib.Path, outputs: List[Tuple[str, int]], input_topics: List[str]):
        self.name: str = name
        self.log_dir: pathlib.Path = log_dir
        self.input_topics: List[str] = input_topics
        self.inputs: Dict[str, Optional[mp.Queue]] = {}
        self.outputs: Dict[str, mp.Queue] = {name: mp.Queue(maxsize=maxsize) for (name, maxsize) in outputs}

    def subscribe(self, topic: str, queue: mp.Queue):
        return self.inputs.update({topic: queue})

    def publish(self, topic: str, data: Any) -> None:
        self.outputs[topic].put(data)

    def get(self, topic: str) -> Any:
        return self.inputs[topic].get()

    def start(self):
        raise NotImplementedError

    def __enter__(self):
        self.logger: logging.Logger = get_logger(f"module_{self.name}", self.log_dir)
        self.logger.info(f"Module {self.name} started.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.warning(f"Module {self.name} is shutting down...")
        self.cleanup()
        self.logger.exception(traceback.format_exc())
        exit()

    def cleanup(self):
        pass

