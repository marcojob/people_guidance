import pathlib
import time
import math
import collections
from typing import Dict
import json

import numpy as np

from ..module import Module
from ...utils import ROOT_DATA_DIR


IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])


def frame_from_input_data(input_data: Dict) -> IMUFrame:
    return IMUFrame(
        ax=float(input_data['data']['accel_x']),
        ay=float(input_data['data']['accel_y']),
        az=float(input_data['data']['accel_z']),
        gx=float(input_data['data']['gyro_x']) * math.pi / 180,
        gy=float(input_data['data']['gyro_y']) * math.pi / 180,
        gz=float(input_data['data']['gyro_z']) * math.pi / 180,
        ts=input_data['timestamp']
    )


class IMUCalibrationModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super().__init__(name="imu_calibration_module",
                         inputs=["drivers_module:accelerations"],
                         log_dir=log_dir)

        self.initial_sleep = 5000  # ms
        self.calibration_duration = 10000  # ms
        self.imu_frames = []

    def start(self):
        self.logger.critical("Starting IMU calibration. Make sure the device is at rest!")
        self.logger.critical(f"Calibration with duration {self.calibration_duration} "
                             f"ms will start after {self.initial_sleep} ms")
        time.sleep(int(self.initial_sleep / 1000))
        self.logger.critical("Calibration started!")
        start_time = self.get_time_ms()
        while self.get_time_ms() - start_time < self.calibration_duration:
            for _ in range(10):
                input_data = self.get("drivers_module:accelerations")
                if not input_data:
                    time.sleep(0.0001)
                else:
                    frame = frame_from_input_data(input_data)
                    self.imu_frames.append(frame)

        duration = self.get_time_ms() - start_time

        statistics: Dict = self.compute_statistics(duration)
        self.save_statistics(statistics)

        self.logger.critical("Finished calibration. Exiting....")

    def compute_statistics(self, duration: float):

        statistics: Dict = {}

        for attr in ("ax", "ay", "az", "gx", "gy", "gz"):
            values = [getattr(frame, attr) for frame in self.imu_frames]
            statistics.update({attr: (np.mean(values), np.var(values))})

        statistics.update({"calibration_duration": duration})

        return statistics

    @staticmethod
    def save_statistics(statistics: Dict):
        with open(ROOT_DATA_DIR / "imu_calibration.json", "w") as fp:
            json.dump(statistics, fp)

