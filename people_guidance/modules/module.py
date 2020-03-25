import multiprocessing as mp
import pathlib
import logging
import traceback
import queue
import time

from typing import Optional, Any, Dict, List, Tuple, Callable, Union

from ..utils import get_logger


class ModuleService:
    def __init__(self, name: str):
        self.name = name
        self.requests = mp.Queue(maxsize=1)
        self.responses = mp.Queue(maxsize=1)
        self.handler = self.default_handler
        self.logger = None

    def register_handler(self, handler: Callable):
        self.handler = handler

    def default_handler(self, request) -> Dict:
        self.logger.warning("Request made to service {self.name} which has no handler. Returning an arbitrary response.")
        return {"id": request["id"], "payload": None}


class Module:
    def __init__(self, name: str, log_dir: pathlib.Path, outputs: List[Tuple[str, int]], input_topics: List[str],
                 services: List[str], requests: List[str]):
        self.name: str = name
        self.log_dir: pathlib.Path = log_dir
        self.input_topics: List[str] = input_topics
        self.inputs: Dict[str, Optional[mp.Queue]] = {}
        self.outputs: Dict[str, mp.Queue] = {name: mp.Queue(maxsize=maxsize) for (name, maxsize) in outputs}

        self.requests: List[str] = requests
        self.request_targets: Dict[str, Dict[str, mp.Queue]] = {}

        self.services: Dict[str, ModuleService] = {name: ModuleService(name) for name in services}

    def subscribe(self, topic: str, queue_obj: mp.Queue):
        return self.inputs.update({topic: queue_obj})

    def add_request_target(self, request_topic, request_queue, response_queue):
        self.request_targets.update({request_topic: {"requests": request_queue, "responses": response_queue}})

    def publish(self, topic: str, data: Any, validity: int, timestamp=None) -> None:
        if timestamp is None:
            # We need to set the timestamp if not set explicitly
            timestamp = self.get_time_ms()

        while True:
            try:
                self.outputs[topic].put_nowait({'data': data, 'timestamp': timestamp, 'validity': validity})
                break
            except queue.Full:
                try:
                    self.outputs[topic].get_nowait()
                except queue.Empty:
                    pass

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

    def make_request(self, target_name, request_payload: Dict):
        full_exc = queue.Full(f"You made another request to {target_name} before it finished the first request."
                              f"You must use await_response to wait for the response first.")
        if not self.request_targets[target_name]["requests"].full():
            try:
                self.request_targets[target_name]["requests"].put_nowait(request_payload)
            except queue.Full:
                raise full_exc
        else:
            raise full_exc

    def await_response(self, target_name) -> Any:
        # this call blocks until a response is received.
        return self.request_targets[target_name]["responses"].get()

    def handle_requests(self):
        for service_name in self.services:
            service = self.services[service_name]
            if not service.requests.empty():
                try:
                    request: Dict = service.requests.get_nowait()
                    self.respond(service_name, request)
                except queue.Empty:
                    pass

    def respond(self, service_name: str, request: Dict):
        full_exc = queue.Full("Response was not read by requesting process. You must read the response in the "
                              "requesting process before making another request.")
        service = self.services[service_name]
        if not service.responses.full():
            try:
                response = {"id": request["id"], "payload": service.handler(request)}
                service.responses.put_nowait(response)
            except queue.Full:
                raise full_exc
        else:
            raise full_exc

    def start(self):
        raise NotImplementedError

    @staticmethod
    def get_time_ms():
        # https://www.python.org/dev/peps/pep-0418/#time-monotonic
        return int(round(time.monotonic() * 1000))

    def __enter__(self):
        self.logger: logging.Logger = get_logger(f"module_{self.name}", self.log_dir)
        for service in self.services.values():
            service.logger = self.logger.getChild(f"service_{service.name}")
        self.logger.info(f"Module {self.name} started.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.warning(f"Module {self.name} is shutting down...")
        self.cleanup()
        self.logger.exception(traceback.format_exc())
        exit()

    def cleanup(self):
        pass
