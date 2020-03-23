import multiprocessing as mp
import pathlib
import logging
import traceback
import queue
import time

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

    def publish(self, topic: str, data: Any, validity: int, timestamp=None) -> None:
        if timestamp is None:
            # We need to set the timestamp if not set explicitly
            timestamp = self.get_time_ms()

        while True:
            try:
                self.outputs[topic].put_nowait({'data': data, 'timestamp': timestamp, 'validity': validity})
                return None
            except queue.Full:
                self.outputs[topic].get_nowait()

    def get(self, topic: str) -> Dict:
        # If the queue is empty we return an empty dict, error handling should be done after

        def is_valid(payload_obj: Dict):
            return payload is not None and payload_obj['timestamp'] + payload_obj['validity'] < self.get_time_ms()

        try:
            payload = None
            while True:
                # get objects from the queue until it is either empty or a valid payload is found.
                # if the queue is empty queue.Empty will be raised.
                payload = self.inputs[topic].get_nowait()
                if is_valid(payload):
                    return payload
        except queue.Empty:
            return dict()

    def start(self):
        raise NotImplementedError

    def get_time_ms(self):
        # https://www.python.org/dev/peps/pep-0418/#time-monotonic
        return int(round(time.monotonic() * 1000))

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
