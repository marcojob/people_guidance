import pathlib
import math
import logging
import time
from typing import Dict, Optional, List
import random

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from ..module import Module
from .config import DEFAULT_CONFIG
from ..drivers_module import ACCEL_G



class Position:
    def __init__(self, timestamp, x, y, z, roll, pitch, yaw):
        super().__init__()
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __getitem__(self, item):
        return self.__dict__[item]

    @staticmethod
    def new_interpolate(timestamp, position0, position1):
        assert position0.timestamp <= timestamp <= position1.timestamp, \
            "timestamp for interpolation must lie between the position timestamps."

        lever = (timestamp - position0.timestamp) / (position1.timestamp - position0.timestamp)

        properties = {"timestamp": timestamp}
        for key in ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"]:
            value = position0[key] + ((position1[key] - position0[key]) * lever)
            properties[key] = value
        return Position(**properties)

    @staticmethod
    def new_empty():
        return Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class IMUDataframe:
    def __init__(self, imu_data):
        self.accel_x = float(imu_data['data']['accel_x'])
        self.accel_y = float(imu_data['data']['accel_y'])
        self.accel_z = float(imu_data['data']['accel_z'])
        self.gyro_x = float(imu_data['data']['gyro_x']) * math.pi / 180
        self.gyro_y = float(imu_data['data']['gyro_y']) * math.pi / 180
        self.gyro_z = float(imu_data['data']['gyro_z']) * math.pi / 180
        self.ts = imu_data['timestamp']


class Velocity:
    def __init__(self, **config):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.dampening_activated = config["dampening_activated"]
        self.dampening_coeffs: List[float] = config["dampening_coeffs"]
        self.dampening_freq = config["dampening_frequency"]
        self.last_dampening = time.monotonic()

    def dampen(self) -> None:
        curr_time = time.monotonic()
        if self.dampening_activated and (curr_time - self.last_dampening) * self.dampening_freq > 1:
            self.x *= self.dampening_coeffs[0]
            self.y *= self.dampening_coeffs[1]
            self.z *= self.dampening_coeffs[2]

    def __str__(self):
        return str((self.x, self.y, self.z))


class PositionEstimator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else DEFAULT_CONFIG

        self.pos = Position.new_empty()
        self.velocity = Velocity(**self.config["velocity"])
        self.prev_frame = None
        self.last_logs: Dict[str, float] = {"accel_rot": time.monotonic()}
        self.last_update = time.monotonic()

    def update(self, imu_data: dict):
        frame = IMUDataframe(imu_data)
        if self.prev_frame is not None:
            self.update_estimation(frame)

        self.prev_frame = frame

    def update_estimation(self, frame: IMUDataframe):
        self.complementary_filter(frame)
        self.position_estimation_simple(frame)
        self.pos.timestamp = frame.ts

    def complementary_filter(self, frame: IMUDataframe):
        dt = frame.ts - self.prev_frame.ts
        alpha = self.config["complementary_filter"]["alpha"]
        # Only integrate for the yaw
        self.pos.yaw += frame.gyro_z * dt
        # Compensate for drift with accelerometer data
        roll_accel = math.atan(frame.accel_y / math.sqrt(frame.accel_x ** 2 + frame.accel_z ** 2))
        pitch_accel = math.atan(frame.accel_x / math.sqrt(frame.accel_y ** 2 + frame.accel_z ** 2))

        # Gyro data
        a = frame.gyro_y * math.sin(self.pos.roll) + frame.gyro_z * math.cos(self.pos.roll)
        roll_vel_gyro = frame.gyro_x + a * math.tan(self.pos.pitch)
        pitch_vel_gyro = frame.gyro_y * math.cos(self.pos.roll) - frame.gyro_z * math.sin(self.pos.roll)
        # Update estimation
        b = self.pos.roll + roll_vel_gyro * dt
        self.pos.roll = (1.0 - alpha) * b + alpha * roll_accel
        c = self.pos.pitch + pitch_vel_gyro * dt
        self.pos.pitch = (1.0 - alpha) * c + alpha * pitch_accel

    def position_estimation_simple(self, frame: IMUDataframe):
        dt = frame.ts - self.prev_frame.ts
        r = Rotation.from_euler('xyz', [self.pos.roll, self.pos.pitch, self.pos.yaw], degrees=True)
        accel_rot = r.apply([frame.accel_x, frame.accel_y, frame.accel_z])

        self.velocity.x += (accel_rot[0] - self.config["acc_correction"]["coeffs"][0]) * dt
        self.velocity.y += (accel_rot[1] - self.config["acc_correction"]["coeffs"][1]) * dt
        # TODO : check coordinate system setup
        self.velocity.z += (accel_rot[2] + ACCEL_G - self.config["acc_correction"]["coeffs"][2]) * dt

        # Reduce the velocity to reduce the drift
        self.velocity.dampen()

        # Integrate to get the position
        self.pos.x += self.velocity.x * dt
        self.pos.y += self.velocity.y * dt
        self.pos.z += self.velocity.z * dt

        self.fixed_frequency_log("accel_rot", f"Accel after rot : {accel_rot}, current speed : {self.velocity}", 0.3)

    def fixed_frequency_log(self, msg_name: str,  msg: str, frequency: float, level=logging.INFO):
        if msg_name not in self.last_logs:
            self.last_logs[msg_name] = time.monotonic()
        if (time.monotonic() - self.last_logs[msg_name]) * frequency > 1:
            logging.info(msg)
            self.last_logs[msg_name] = time.monotonic()



class PositionEstimationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module", outputs=[("position_vis", 10)],
                                                       inputs=["drivers_module:accelerations"],
                                                       services=["position_request"],
                                                       log_dir=log_dir)

        self.estimator = PositionEstimator()
        self.visu_frequency = 10
        self.last_visu_publish = time.monotonic()

    def start(self):

        self.services["position_request"].register_handler(self.handle_position_request)

        expected_input_fps = 80.0
        update_count = 0
        fps = 0.0
        start_time = time.time()

        target_fps = 60
        dropout_proba = 0.5
        while True:
            # request = self.get_requests("position_request")
            self.handle_requests()
            input_data = self.get("drivers_module:accelerations")
            if input_data:
                update_count += 1

                self.estimator.update(input_data)

                if update_count > 200:
                    curtime = time.time()
                    fps = update_count / (curtime - start_time)
                    self.logger.info(f"updating at {fps} with dropout {dropout_proba}")
                    update_count = 0
                    start_time = curtime


    def publish_visu_data(self):
        if (time.monotonic() - self.last_visu_publish) * self.visu_frequency > 1:
            self.publish("position_vis", self.estimator.pos.__dict__, 200, self.estimator.prev_frame.ts)

    def handle_position_request(self, request):
        timestamp = request["payload"]

        self.logger.info(f"Got request with offset {timestamp - self.estimator.pos.timestamp} ")
        return {"id": request["id"], "payload": self.estimator.pos.__dict__}