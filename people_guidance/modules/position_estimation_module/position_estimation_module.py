import io
import platform
import re
import math
import logging
from time import sleep, monotonic, perf_counter
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

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])

# TODO: IMU data (acceleration) needs to be multiplied by (-1) to compensate for calculation error
# TODO: Remove the hardcoded change in this file for further datasets recorded after the correction.
HARDCODED_NEGATIVE_CORRECTION = -1
''' 
Coordinates valid for the IMU only : looking at the front of the camera,  
x points upwards, y horizontally to the right, z horizontally towards the camera, 
in direction of the plane
'''
# Position estimation based on the data from the accelerometer and the gyroscope
class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module", outputs=[("position", 10)],
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
                    self.prev_imu_frame = frame
                else:
                    self.update_position(frame)
                    # TODO: do not track and do not update if the timestamp did not change (dt = 0)
                    # ERROR description : 91426359 appears twice in the printed list.
                    self.append_tracked_positions()
                    # Display in a scatter plot
                    self.display_position(frame)

            self.publish_to_visualization()

    @staticmethod
    def frame_from_input_data(input_data: Dict) -> IMUFrame:

        return IMUFrame(
            ax=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_x']),
            ay=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_y']),
            az=HARDCODED_NEGATIVE_CORRECTION * float(input_data['data']['accel_z']),
            gx=float(input_data['data']['gyro_x']) * math.pi / 180,
            gy=float(input_data['data']['gyro_y']) * math.pi / 180,
            gz=float(input_data['data']['gyro_z']) * math.pi / 180,
            ts=input_data['timestamp']
        )

    def append_tracked_positions(self):
        self.tracked_positions.append(copy.deepcopy(self.pos))

        if len(self.tracked_positions) > 300:
            self.tracked_positions.pop(0)

    def display_position(self, frame):
        # place figure in top left corner
        plt.figure(1, figsize=(10, 5))
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(5, 5, 1000, 1000)

        plt.subplot(2, 3, 1)
        plt.scatter(self.pos.x, self.pos.y)
        plt.title('Position estimation')
        plt.suptitle(f'Parameters : \n'
                     f'METHOD_RESET_VELOCITY: {METHOD_RESET_VELOCITY}, RESET_VEL_FREQ : {RESET_VEL_FREQ}, \n'
                     f'RESET_VEL_FREQ_COEF_X : {RESET_VEL_FREQ_COEF_X}, RESET_VEL_FREQ_COEF_Y : {RESET_VEL_FREQ_COEF_Y}, RESET_VEL_FREQ_COEF_Z : {RESET_VEL_FREQ_COEF_Z}, \n'
                     f'METHOD_ERROR_ACC_CORRECTION : {METHOD_ERROR_ACC_CORRECTION}, CORRECTION_ACC : {CORRECTION_ACC}')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        #
        plt.subplot(2, 3, 2)
        plt.scatter(self.pos.x, self.pos.z)
        plt.title('Position estimation')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        #
        plt.subplot(2, 3, 3)
        plt.scatter(self.pos.roll, self.pos.pitch)
        plt.title('Angle')
        plt.xlabel('roll [rad]')
        plt.ylabel('pitch [rad]')
        #
        plt.subplot(2, 3, 4)
        plt.scatter(self.pos.roll, self.pos.yaw)
        plt.title('Angle')
        plt.xlabel('roll [rad]')
        plt.ylabel('yaw [rad]')
        #
        plt.subplot(2, 3, 5)
        plt.scatter(frame.ax, frame.ay)
        plt.title('Acceleration')
        plt.xlabel('x [acc]')
        plt.ylabel('y [acc]')
        #
        plt.subplot(2, 3, 6)
        plt.scatter(frame.ax, frame.az)
        plt.title('Acceleration')
        plt.xlabel('x [acc]')
        plt.ylabel('z [acc]')
        #
        plt.pause(0.0001)
        # plt.show()

    def update_position(self, frame: IMUFrame) -> None:
        dt: float = (frame.ts - self.prev_imu_frame.ts) / 1000.0
        self.pos.ts = frame.ts
        self.complementary_filter(frame, dt)
        self.position_estimation_simple(frame, dt)
        self.prev_imu_frame = frame

    def reset_drift_tracker(self):
        return {
            "total_time": 0.0,
            "n_elt_summed": 0,
            "toal_acc_x": 0,
            "toal_acc_y": 0,
            "toal_acc_z": 0
        }

    def complementary_filter(self, frame: IMUFrame, dt: float):
        # Only integrate for the yaw
        self.pos.yaw += frame.gz * dt
        # Compensate for drift with accelerometer data
        roll_accel = math.atan(frame.ay / math.sqrt(frame.ax ** 2 + frame.az ** 2))
        pitch_accel = math.atan(frame.ax / math.sqrt(frame.ay ** 2 + frame.az ** 2))
        # Gyro data
        a = frame.gz * math.sin(self.pos.roll) + frame.gz * math.cos(self.pos.roll)
        roll_vel_gyro = frame.gz + a * math.tan(self.pos.pitch)
        pitch_vel_gyro = frame.gy * math.cos(self.pos.roll) - frame.gz * math.sin(self.pos.roll)
        # Update estimation
        b = self.pos.roll + roll_vel_gyro * dt
        self.pos.roll = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * b + ALPHA_COMPLEMENTARY_FILTER * roll_accel
        c = self.pos.pitch + pitch_vel_gyro * dt
        self.pos.pitch = (1.0 - ALPHA_COMPLEMENTARY_FILTER) * c + ALPHA_COMPLEMENTARY_FILTER * pitch_accel
        self.logger.info(f"updated timestamp {frame.ts}")

    def position_estimation_simple(self, frame: IMUFrame, dt: float):
        # Integrate the acceleration after transforming the acceleration in world coordinates
        r = R.from_euler('xyz', [self.pos.roll, self.pos.pitch, self.pos.yaw], degrees=True)
        accel_rot = r.apply([frame.ax, frame.ay, frame.az])
        self.speed["x"] += (accel_rot[0] - CORRECTION_ACC[0]) * dt
        self.speed["y"] += (accel_rot[1] - CORRECTION_ACC[1]) * dt
        self.speed["z"] += (accel_rot[2] + ACCEL_G - CORRECTION_ACC[2]) * dt  # TODO : check coordinate system setup
        # Calculate the mean for drift compensation
        if MEASURE_SUMMED_ERROR_ACC:
            self.drift_tracking["total_time"] += dt
            self.drift_tracking["n_elt_summed"] += 1
            self.drift_tracking["toal_acc_x"] += accel_rot[0]
            self.drift_tracking["toal_acc_y"] += accel_rot[1]
            self.drift_tracking["toal_acc_z"] += accel_rot[2] + ACCEL_G

            self.const_frequency_log("log_drift_stats", PUBLISH_SUMMED_MEASURE_ERROR_ACC,
                                     f"Drift {self.drift_tracking}",
                                     )

        # Reduce the velocity to reduce the drift
        self.dampen_velocity()

        # Integrate to get the position
        self.pos.x += self.speed["x"] * dt
        self.pos.y += self.speed["y"] * dt
        self.pos.z += self.speed["z"] * dt

        self.const_frequency_log("last_summed_acc", POSITION_PUBLISH_ACC_FREQ,
                                 f"Acceleration after rotation :  {accel_rot}, "
                                 f"Corrected speed : {[self.speed['x'], self.speed['y'], self.speed['z']]} ",
                                 )

    def dampen_velocity(self):
        if METHOD_RESET_VELOCITY:
            curr_time = monotonic()
            if "velocity_reset" not in self.event_timestamps or \
                    (curr_time - self.event_timestamps["velocity_reset"]) * RESET_VEL_FREQ > 1:
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
                (curr_time - self.event_timestamps[msg_name]) * frequency > 1:
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
