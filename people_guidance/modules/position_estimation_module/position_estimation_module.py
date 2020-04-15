import io
import platform
import re
import math
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

# TODO: IMU data (acceleration) needs to be multiplied by (-1) to compensate for calculation error
# TODO: Remove the hardcoded change in this file for further datasets recorded after the correction.
HARDCODED_NEGATIVE_CORRECTION = 1

''' 
Coordinates valid for the IMU only : looking at the front of the camera,  x points upwards, 
y horizontally to the right, z horizontally towards the camera, towards the plane.
Output in camera coordinate system: looking at the front of the camera,  z points upwards, 
y horizontally to the right, x horizontally from the camera, out of the plane.
In Camera coordinates: Z = X_IMU, X = -Z_IMU, Y = Y_IMU (-90° rotation around the Y axis)
'''


# TODO: gravity compensation occurs in the starting frame. Averaging the first few elements for defining precisely the
#  direction of the gravity and setting the reference cordinate frame can help

class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module",
                                                       outputs=[("position_vis", 10), ("position", 10)],
                                                       inputs=["drivers_module:accelerations"],
                                                       services=["position_request"],
                                                       log_dir=log_dir)
        self.args = args

        self.drift_tracking: Dict = self.reset_drift_tracker()
        self.pos: Position = new_empty_position()
        self.tracked_positions: List[Position] = []
        self.prev_imu_frame: Optional[IMUFrame] = None
        self.last_visualized_pos: Optional[Position] = None

        self.event_timestamps: Dict[str, float] = {}  # keep track of when events (classified by name) happened last
        self.speed: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.acceleration: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}

    def start(self):
        # TODO: evaluate position quality
        # TODO: save last few estimations with absolute timestamp
        self.services["position_request"].register_handler(self.position_request)

        while True:
            input_data = self.get("drivers_module:accelerations")
            self.handle_requests()

            if not input_data:  # m/s^2 // °/s
                sleep(0.0001)
            else:
                frame = self.frame_from_input_data(input_data)

                if self.prev_imu_frame is None:
                    # if the frame we just received is the first one we have received.
                    self.prev_imu_frame = frame  # TODO: check where used. should use rotated elements instead?
                else:
                    # TODO: do not track and do not update if the timestamp did not change (dt = 0)
                    # ERROR description : 91426359 appears twice in the printed list.
                    self.update_position(frame)
                    self.append_tracked_positions()

                    # Display in a scatter plot (debug)
                    visualize_locally(self.pos, frame, self.drift_tracking, self.acceleration, plot_pos=True,
                                      plot_angles=True, plot_acc_input=True, plot_acc_transformed=True)

            self.publish_to_visualization()

    @staticmethod
    def frame_from_input_data(input_data: Dict) -> IMUFrame:
        # In Camera coordinates: Z = X_IMU, X = -Z_IMU, Y = Y_IMU (-90° rotation around the Y axis)
        return IMUFrame(
            ax=HARDCODED_NEGATIVE_CORRECTION * -float(input_data['data']['accel_z']),
            ay=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_y']),
            az=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_x']),
            gx=-float(input_data['data']['gyro_z']) * math.pi / 180,
            gy=float(input_data['data']['gyro_y']) * math.pi / 180,
            gz=float(input_data['data']['gyro_x']) * math.pi / 180,
            ts=input_data['timestamp']
        )

    def append_tracked_positions(self):
        self.tracked_positions.append(copy.deepcopy(self.pos))

        if len(self.tracked_positions) > 300:
            self.tracked_positions.pop(0)

    def update_position(self, frame: IMUFrame) -> None:
        dt: float = (frame.ts - self.prev_imu_frame.ts) / 1000.0
        self.pos.ts = frame.ts
        self.complementary_filter(frame, dt)
        self.position_estimation_simple(frame, dt)
        self.prev_imu_frame = frame

    @staticmethod
    def reset_drift_tracker():
        return {
            "total_time": 0.0,
            "n_elt_summed": 0,
            "total_acc_x": 0,
            "total_acc_y": 0,
            "total_acc_z": 0,
            "total_acc_avg_x": 0,
            "total_acc_avg_y": 0,
            "total_acc_avg_z": 0
        }

    def complementary_filter(self, frame: IMUFrame, dt: float):  # TODO: check formulas
        # Only integrate for the yaw
        self.pos.yaw += frame.gz * dt
        # Compensate for drift with accelerometer data https: // philsal.co.uk / projects / imu - attitude - estimation
        roll_accel = math.atan(frame.ay / math.sqrt(frame.ax ** 2 + frame.az ** 2)) # scalar to rad
        pitch_accel = math.atan(frame.ax / math.sqrt(frame.ay ** 2 + frame.az ** 2)) # scalar to rad
        # Gyro data
        a = frame.gz * math.sin(self.pos.roll) + frame.gz * math.cos(self.pos.roll)
        roll_vel_gyro = frame.gz + a * math.tan(self.pos.pitch)
        pitch_vel_gyro = frame.gy * math.cos(self.pos.roll) - frame.gz * math.sin(self.pos.roll)
        # Update estimation

        b = self.pos.roll + roll_vel_gyro * dt
        self.pos.roll = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * b + ALPHA_COMPLEMENTARY_FILTER * roll_accel
        c = self.pos.pitch + pitch_vel_gyro * dt
        self.pos.pitch = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * c + ALPHA_COMPLEMENTARY_FILTER * pitch_accel

    def position_estimation_simple(self, frame: IMUFrame, dt: float):
        # Integrate the acceleration after transforming the acceleration in world coordinates
        self.acceleration_after_rotation_gravity_compensation(frame)
        self.speed["x"] += (self.acceleration["x"] - CORRECTION_ACC[0]) * dt
        self.speed["y"] += (self.acceleration["y"] - CORRECTION_ACC[1]) * dt
        self.speed["z"] += (self.acceleration["z"] - CORRECTION_ACC[2]) * dt

        # Measure the sum of the deviation from 0
        self.drift_measurement(dt)

        # Reduce the velocity to reduce the drift
        self.dampen_velocity()

        # Integrate to get the position
        self.pos.x += self.speed["x"] * dt
        self.pos.y += self.speed["y"] * dt
        self.pos.z += self.speed["z"] * dt

        self.const_frequency_log("new_element_and_timestamp", POSITION_PUBLISH_NEW_TIMESTAMP,
                                 f"Position updated at timestamp: {frame.ts}, time interval: {dt}",
                                 )
        self.const_frequency_log("last_summed_acc", POSITION_PUBLISH_ACC_FREQ,
                                 f"Acceleration after rotation :  {[self.acceleration['x'], self.acceleration['y'], self.acceleration['z']]}, "
                                 f"Corrected speed : {[self.speed['x'], self.speed['y'], self.speed['z']]} ",
                                 )

    def acceleration_after_rotation_gravity_compensation(self, frame: IMUFrame):
        r = R.from_euler('xyz', [self.pos.roll, self.pos.pitch, self.pos.yaw], degrees=True)
        # rotation to starting coordinates and gravity compensation
        [self.acceleration["x"], self.acceleration["y"], self.acceleration["z"]] = \
            r.apply([frame.ax, frame.ay, frame.az]) - [0, 0, ACCEL_G]

    def drift_measurement(self, dt: float):  # TODO: add type Vorschlag
        # Calculate the mean for drift compensation
        if MEASURE_SUMMED_ERROR_ACC:
            self.drift_tracking["total_time"] += dt
            self.drift_tracking["n_elt_summed"] += 1
            self.drift_tracking["total_acc_x"] += self.acceleration["x"]
            self.drift_tracking["total_acc_y"] += self.acceleration["y"]
            self.drift_tracking["total_acc_z"] += self.acceleration["z"]
            self.drift_tracking["total_acc_avg_x"] = round(
                self.drift_tracking["total_acc_x"] / self.drift_tracking["n_elt_summed"], 3)
            self.drift_tracking["total_acc_avg_y"] = round(
                self.drift_tracking["total_acc_y"] / self.drift_tracking["n_elt_summed"], 3)
            self.drift_tracking["total_acc_avg_z"] = round(
                self.drift_tracking["total_acc_z"] / self.drift_tracking["n_elt_summed"], 3)

            self.const_frequency_log("log_drift_stats", PUBLISH_SUMMED_MEASURE_ERROR_ACC,
                                     f"Drift {self.drift_tracking}")

    def dampen_velocity(self):  # TODO : reset dependent on dt
        if METHOD_RESET_VELOCITY:
            curr_time = monotonic()
            if "velocity_reset" not in self.event_timestamps or \
                    (curr_time - self.event_timestamps["velocity_reset"]) * RESET_VEL_FREQ > 1 or \
                    RESET_VEL_FREQ > 99:
                self.speed["x"] *= RESET_VEL_FREQ_COEF_X
                self.speed["y"] *= RESET_VEL_FREQ_COEF_Y
                self.speed["z"] *= RESET_VEL_FREQ_COEF_Z

                self.event_timestamps["velocity_reset"] = curr_time

    def publish_to_visualization(self):
        curr_time = monotonic()
        if "last_visu_pub" not in self.event_timestamps or self.last_visualized_pos is None or \
                (curr_time - self.event_timestamps["last_visu_pub"]) * POSITION_PUBLISH_FREQ > 1:

            if self.pos != self.last_visualized_pos:
                self.publish("position", self.pos.__dict__, POS_VALIDITY_MS)
                self.last_visualized_pos = copy.deepcopy(self.pos)
                self.event_timestamps["last_visu_pub"] = curr_time

    def const_frequency_log(self, msg_name: str, frequency: int, msg: str, level=logging.INFO):
        curr_time = monotonic()
        if msg_name not in self.event_timestamps or \
                (curr_time - self.event_timestamps[msg_name]) * frequency > 1 or \
                frequency > 99:
            self.logger.info(msg)
            self.event_timestamps[msg_name] = curr_time

    def position_request(self, request):

        requested_timestamp = request["payload"]
        neighbors = [None, None]

        self.logger.info(f"{len(self.tracked_positions)}  - {[str(pos.ts) for pos in self.tracked_positions]}")

        for idx, position in enumerate(self.tracked_positions):
            # this assumes that our positions are sorted old to new.
            if position.ts <= requested_timestamp:
                neighbors[0] = (idx, position)
            if position.ts >= requested_timestamp:
                neighbors[1] = (idx, position)
                break

        if neighbors[0] is not None and neighbors[1] is not None:
            interp_position = new_interpolated_position(requested_timestamp, neighbors[0][1], neighbors[1][1])
            return interp_position.__dict__

        elif neighbors[0] is None and neighbors[1] is not None and neighbors[1][0] < len(self.tracked_positions) - 1:
            self.logger.critical("Extrapolating backwards!")
            interp_position = new_interpolated_position(requested_timestamp, neighbors[1][1],
                                                        self.tracked_positions[neighbors[1][0] + 1])
            return interp_position.__dict__

        else:
            offset = self.pos.ts - requested_timestamp
            self.logger.info(f"Could not interpolate for position with timestamp {requested_timestamp}."
                             f"Current offset {offset}")
            return None
