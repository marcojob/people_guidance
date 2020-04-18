import multiprocessing as mp
import time
import logging
import pathlib
import datetime
from typing import Callable, Optional, List, Dict, Tuple
from psutil import cpu_percent, virtual_memory

from .utils import get_logger, ROOT_LOG_DIR, init_logging
from .modules import Module


class Pipeline:

    def __init__(self, args=None):
        self.log_dir: pathlib.Path = self.create_log_dir()
        self.logger: logging.Logger = get_logger("pipeline", self.log_dir)
        self.modules: Dict[Module] = {}
        self.processes: List[mp.Process] = []
        self.args = args

    def start(self):
        self.connect_subscriptions()
        self.connect_services()
        with self:
            for module in self.modules.values():
                p = mp.Process(target=self.start_module, kwargs={"module": module}, daemon=True)
                self.processes.append(p)
                p.start()

            while True:
                time.sleep(2)
                if not all([proc.is_alive() for proc in self.processes]):
                    self.logger.exception("Found dead child process. Pipeline will terminate all children and exit.")
                    exit()
                else:
                    self.logger.info(f"Pipeline alive: CPU: {cpu_percent()}, Memory: {virtual_memory()._asdict()['percent']}")

                # try:
                #     for module in self.modules.values():
                #         for input_name, input_queue in module.inputs.items():
                #             logging.debug(f"{module.name} input {input_name} queue size {input_queue.qsize()}")
                #         for output_name, output_queue in module.outputs.items():
                #             logging.debug(f"{module.name} output {output_name} queue size {output_queue.qsize()}")
                # except NotImplementedError:
                #     self.logger.debug("Could not load queue size because the platform does not support it.")

    @staticmethod
    def start_module(module: Module):
        init_logging()
        with module:
            module.start()

    def add_module(self, constructor: Callable, log_level=logging.DEBUG):
        module = constructor(log_dir=self.log_dir, args=self.args)
        module.log_level = log_level
        if module.name in self.modules:
            raise RuntimeError(f"Could not create a module with name {module.name} "
                               "because another module had the same name. Module names must be unique!")
        self.modules.update({module.name: module})

    def connect_subscriptions(self):
        for module in self.modules.values():
            try:
                for channel_name in module.inputs:
                    channel = self.get_channel(channel_name)
                    module.subscribe(channel_name, channel)
            except KeyError:
                raise KeyError(f"Could not subscribe module {module.name}")

    def connect_services(self):
        for module in self.modules.values():
            try:
                for request_channel in module.requests:
                    request_queue, response_queue = self.get_service(request_channel)
                    module.add_request_target(request_channel, request_queue, response_queue)
            except KeyError:
                raise KeyError(f"Could not link service for module {module.name}")

    def get_service(self, request_channel) -> Tuple[mp.Queue, mp.Queue]:
        module_name, service_name = request_channel.split(":")
        if module_name not in self.modules:
            raise KeyError(
                f"Cannot link request {request_channel}: Unknown module {module_name}. Must be one of {self.modules.keys()}")

        services = self.modules[module_name].services
        if service_name not in services:
            raise KeyError(f"Cannot link request {request_channel}: Unknown service {service_name} in "
                           f"module {module_name}. Must be one of {services.keys()}")
        return services[service_name].requests, services[service_name].responses

    def get_channel(self, channel_name):
        module_name, output_name = channel_name.split(":")
        if module_name not in self.modules:
            raise KeyError(
                f"Cannot subscribe to {channel_name}: Unknown module {module_name}. Must be one of {self.modules.keys()}")

        outputs = self.modules[module_name].outputs
        if output_name not in outputs:
            raise KeyError(f"Cannot subscribe to {channel_name}: Unknown output {output_name}. "
                           f"Must be one of {outputs.keys()}")
        return outputs[output_name]

    @staticmethod
    def create_log_dir() -> pathlib.Path:
        time_str = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        log_dir = ROOT_LOG_DIR / time_str
        log_dir.mkdir(parents=True, exist_ok=False)
        return log_dir

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.warning("Terminating all children...")
        for proc in self.processes:
            proc.join(timeout=2)
            if proc.is_alive():
                self.logger.critical("Child did not exit fast enough and will be terminated.")
                proc.kill()



