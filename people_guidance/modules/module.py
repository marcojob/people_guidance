import multiprocessing as mp
import pathlib
import logging
import traceback
import time
import queue

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
        # We need to set the timestamp if not set explicitly
        if timestamp is None:
            timestamp = self.get_time_ms()

        # If the queue is full we need to clear one slot for newer data
        if self.outputs[topic].full():
            self.outputs[topic].get()
            #self.logger.warning( f"Output queue {topic} of {self.name} is full!")

        # In any case add the item to the queue
        self.outputs[topic].put({'data': data, 'timestamp': timestamp, 'validity': validity}, timeout=0)

    def get(self, topic: str) -> Dict:
        # If the queue is empty we return an empty dict, error handling should be done after
        if self.inputs[topic].empty():
            # self.logger.warning(f"Input queue {topic} of {self.name} is empty!")
            return dict()

        # Go through the queue until you find data that is still valid
        while not self.inputs[topic].empty():
            out = self.get_from_queue_nowait(self.inputs[topic])
            if out is not None:
                # If data is valid return it
                if out['timestamp'] + out['validity'] < self.get_time_ms():
                    # self.logger.warning(f"{topic} data not valid anymore")
                    return out
                else:
                    return out

        # The queue is officially empty and nothing is valid :(

        return dict()

    @staticmethod
    def get_from_queue_nowait(queue_obj: mp.Queue) -> Optional[Dict]:
        try:
            return queue_obj.get(block=False)
        except queue.Empty:
            return None

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
