import io
import platform
import re
from math import tan, atan2, cos, sin, pi, sqrt, atan
import logging
from time import sleep, monotonic
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import List, Optional, Dict
from queue import Queue
from pathlib import Path
import copy
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .utils import *
from ..drivers_module import ACCEL_G
from ..module import Module
from ...utils import DEFAULT_DATASET
from .position import Position, new_empty_position, new_interpolated_position

RAD_TO_DEG = 180.0/pi
DEG_TO_RAD = pi/180.0
LP_LEN = 100
ACCEL_G = -9.80600

class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module",
                                                       outputs=[("position_vis", 10)],
                                                       inputs=["drivers_module:accelerations"],
                                                       services=["position_request"],
                                                       log_dir=log_dir)
        self.args = args

        self.pos: Position = new_empty_position()
        self.tracked_positions: List[Position] = []
        self.prev_imu_frame: Optional[IMUFrame] = None
        self.last_visualized_pos: Optional[Position] = None

        self.event_timestamps: Dict[str, float] = {}  # keep track of when events (classified by name) happened last
        self.speed: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.acceleration: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0, "x_last": 0.0, "y_last": 0.0, "z_last": 0.0}

        self.counter_same_request = 0

    def start(self):
        # Contains window of raw accel data
        self.data_raw = {var: list() for var in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]}

        # Low pass filtered sensor values
        self.accel_x_lp = 0.0 # IMU Frame low pass filtered accel x data
        self.accel_y_lp = 0.0 # IMU Frame low pass filtered accel y data
        self.accel_z_lp = 0.0 # IMU Frame low pass filtered accel z data
        self.gyro_x_lp = 0.0 # IMU Frame low pass filtered gyro x data
        self.gyro_y_lp = 0.0 # IMU Frame low pass filtered gyro y data
        self.gyro_z_lp = 0.0 # IMU Frame low pass filtered gyro z data

        self.services["position_request"].register_handler(self.position_request)

        while True:
            input_data = self.get("drivers_module:accelerations")
            if not input_data:
                sleep(0.0001)
            else:
                frame = self.frame_from_input_data(input_data)  # m/s^2 // °/s to rad/s

                if self.prev_imu_frame is None:
                    # if the frame we just received is the first one we have received.
                    self.prev_imu_frame = frame  # TODO: check where used. should use rotated elements instead?
                else:
                    # Calculate dt
                    dt = (frame.ts - self.prev_imu_frame.ts) / 1000.0
                    self.pos.ts = frame.ts

                    # Low pass filter input data
                    self.apply_low_pass_filters(frame)

                    # Apply complementary filter
                    self.complementary_filter(frame, dt)

                    # Estimate position
                    self.position_estimation_simple(frame, dt)

                    # Publish to visualizer
                    self.publish("position_vis", self.pos.__dict__, POS_VALIDITY_MS)

                    # Save last frame
                    self.prev_imu_frame = frame

                    # Keep track of computed positions
                    self.track_positions()

                    # Display in a scatter plot (debug)
                    self.visualize_locally()

            self.handle_requests()

    def apply_low_pass_filters(self, frame):
        # Low pass filter accel data
        self.accel_x_lp = self.low_pass(frame.ax, self.data_raw["accel_x"])
        self.accel_y_lp = self.low_pass(frame.ay, self.data_raw["accel_y"])
        self.accel_z_lp = self.low_pass(frame.az, self.data_raw["accel_z"])

        # Low pass filter gyro data
        self.gyro_x_lp = self.low_pass(frame.gx, self.data_raw["gyro_x"], is_degrees=True)
        self.gyro_y_lp = self.low_pass(frame.gy, self.data_raw["gyro_y"], is_degrees=True)
        self.gyro_z_lp = self.low_pass(frame.gz, self.data_raw["gyro_z"], is_degrees=True)

    def complementary_filter(self, frame: IMUFrame, dt: float):  # TODO: check formulas
        # Pitch and roll based on accel
        pitch_accel = atan(self.accel_y_lp / sqrt(self.accel_z_lp**2 + self.accel_x_lp**2))
        roll_accel = atan(self.accel_z_lp / sqrt(self.accel_y_lp**2 + self.accel_x_lp**2))

        # Pitch, roll and yaw based on gyro
        roll_gyro = self.gyro_z_lp + \
                     self.gyro_y_lp*sin(self.pos.pitch)*tan(self.pos.roll) + \
                     self.gyro_x_lp*cos(self.pos.pitch)*tan(self.pos.roll)

        pitch_gyro = self.gyro_y_lp*cos(self.pos.pitch) - self.gyro_x_lp*sin(self.pos.pitch)

        yaw_gyro = self.gyro_y_lp*sin(self.pos.pitch)*1.0/cos(self.pos.roll) + self.gyro_x_lp*cos(self.pos.pitch)*1.0/cos(self.pos.roll)

        # Apply complementary filter
        self.pos.pitch = (1.0 - ALPHA_CF)*(self.pos.pitch + pitch_gyro*dt) + ALPHA_CF * pitch_accel
        self.pos.roll = (1.0 - ALPHA_CF)*(self.pos.roll + roll_gyro*dt) + ALPHA_CF * roll_accel
        self.pos.yaw += yaw_gyro*dt

    def position_estimation_simple(self, frame, dt: float):
        # Obtain rotation matrix
        r = R.from_euler('xyz', [-self.pos.roll, -self.pos.pitch, -self.pos.yaw], degrees=False)

        # Rotate accelerations to world coordinate system
        accel_w = r.apply([frame.az, frame.ay, frame.ax])

        # Integrate to get the position
        self.pos.x += 0.5*accel_w[0] * dt * dt
        self.pos.y += 0.5*accel_w[1] * dt * dt
        self.pos.z += 0.5*(accel_w[2] - ACCEL_G)* dt * dt

    def position_request(self, request): # TODO: check timeout errors
        requested_timestamp = request["payload"]
        neighbors = [None, None]

        self.logger.debug(f"{len(self.tracked_positions)}  - {[str(pos.ts) for pos in self.tracked_positions]}")
        for idx, position in enumerate(self.tracked_positions):
            # this assumes that our positions are sorted old to new.
            if position.ts <= requested_timestamp:
                neighbors[0] = (idx, position)
            if position.ts >= requested_timestamp:
                neighbors[1] = (idx, position)
                break

        if neighbors[0] is not None and neighbors[1] is not None:
            self.counter_same_request = 0
            self.logger.debug(f"Normal interpolation, asking for ts {requested_timestamp}")
            interp_position = new_interpolated_position(requested_timestamp, neighbors[0][1], neighbors[1][1])
            return interp_position.__dict__

        elif neighbors[0] is None and neighbors[1] is not None and neighbors[1][0] < len(self.tracked_positions) - 1:
            self.counter_same_request = 0
            self.logger.critical("Extrapolating backwards!")
            interp_position = new_interpolated_position(requested_timestamp, neighbors[1][1],
                                                        self.tracked_positions[neighbors[1][0] + 1])
            return interp_position.__dict__

        else:
            offset = self.pos.ts - requested_timestamp
            self.logger.info(f"Could not interpolate for position with timestamp {requested_timestamp}. "
                             f"Current offset {offset}, asked {self.counter_same_request} times")
            if self.counter_same_request > 200: #TODO: REMOVE Quick fix
                self.logger.warning(f"Not interpolating. Quick fix. requested {requested_timestamp}, "
                                    f"sending {self.tracked_positions[-1]}")
                self.counter_same_request = 0
                return self.tracked_positions[-1]
            self.counter_same_request += 1
            return None

    @staticmethod
    def frame_from_input_data(input_data: Dict) -> IMUFrame:
        # In Camera coordinates: Z = X_IMU, X = -Z_IMU, Y = Y_IMU (-90° rotation around the Y axis)
        return IMUFrame(
            ax=float(input_data['data']['accel_x']),
            ay=float(input_data['data']['accel_y']),
            az=float(input_data['data']['accel_z']),
            gx=float(input_data['data']['gyro_x']),
            gy=float(input_data['data']['gyro_y']),
            gz=float(input_data['data']['gyro_z']),
            ts=input_data['data']['timestamp']
        )

    def visualize_locally(self):
        curr_time = monotonic()
        msg_name = "last_visualization"
        if not MEASURE_SUMMED_ERROR_ACC_AUTO or (self.drift_tracking["auto_state"] and MEASURE_SUMMED_ERROR_ACC_AUTO):
            if VISUALIZE_LOCALLY and (msg_name not in self.event_timestamps or \
                    (curr_time - self.event_timestamps[msg_name]) * VISUALIZE_LOCALLY_FREQ > 1 or \
                    VISUALIZE_LOCALLY_FREQ > 99):
                self.logger.info(f"Visualization at current time {curr_time} for frame {frame}")
                self.event_timestamps[msg_name] = curr_time
                visualize_locally(self.pos, frame, self.drift_tracking, self.acceleration, plot_pos=True,
                                    plot_angles=True, plot_acc_input=True, plot_acc_transformed=True)

    def track_positions(self):
        self.tracked_positions.append(copy.deepcopy(self.pos))
        if len(self.tracked_positions) > 300:
            self.tracked_positions.pop(0)

    def low_pass(self, val, data_raw, max_len=LP_LEN, is_degrees=False):
        val = float(val)

        # If it is in degrees, convert to radians
        if is_degrees:
            val = val*DEG_TO_RAD

        # Only keep LP_LEN data points
        if len(data_raw) > max_len:
            del data_raw[0]
        data_raw.append(val)

        # Calculate mean of current window
        val_lp = 0.0
        for v in data_raw:
            val_lp += v
        val_lp /= len(data_raw)

        return val_lp