import io
import platform
import re
import math
from time import sleep, monotonic, perf_counter
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import List, Optional, Dict
from queue import Queue
from pathlib import Path
import copy
import collections

from .utils import *
from ..drivers_module import ACCEL_G
from ..module import Module
from ...utils import DEFAULT_DATASET
from .position import Position, new_empty_position, new_interpolated_position

IMUFrame = collections.namedtuple("IMUFrame", ["ax", "ay", "az", "gx", "gy", "gz", "ts"])


# Position estimation based on the data from the accelerometer and the gyroscope
class PositionEstimationModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(PositionEstimationModule, self).__init__(name="position_estimation_module", outputs=[("position", 10)],
                                                       inputs=["drivers_module:accelerations"],
                                                       services=["position_request"],
                                                       log_dir=log_dir)
        self.args = args

        # General inits
        self.count_valid_input = 0  # Number of inputs processed
        self.countall = 0  # Number of loops since start
        self.count_outputs = 0  # Number of elements published
        # Timestamps
        self.timestamp_last_input = 0  # time last input received
        self.timestamp_last_output = monotonic()  # time last output published
        self.timestamp_last_displayed_input = monotonic()
        self.timestamp_last_displayed_acc = monotonic()
        self.timestamp_last_reset_vel = monotonic()
        self.timestamp_last_summed_acc = monotonic()
        self.loop_time = monotonic()  # time start of loop

        # Initialization and tracking
        self.dt_initialised = False
        self.last_data_dict_published = None
        # Tracking drift
        self.total_time = 0
        self.total_number_elt_summed = 0
        self.total_acc_x = 0
        self.total_acc_y = 0
        self.total_acc_z = 0


        # Output_speed (m/s)
        self.speed_x = 0.
        self.speed_y = 0.
        self.speed_z = 0.
        # Timestamp element

        self.pos: Position = new_empty_position()
        self.tracked_positions: List[Position] = []
        self.prev_imu_frame: Optional[IMUFrame] = None
        self.last_visualized_pos: Optional[Position] = None

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
                    self.append_tracked_positions()

            self.publish_to_visualization()


    @staticmethod
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
        self.speed_x += (accel_rot[0] - CORRECTION_ACC[0]) * dt
        self.speed_y += (accel_rot[1] - CORRECTION_ACC[1]) * dt
        self.speed_z += (accel_rot[2] + ACCEL_G - CORRECTION_ACC[2]) * dt  # TODO : check coordinate system setup
        # Calculate the mean for drift compensation
        if MEASURE_SUMMED_ERROR_ACC:
            self.total_time += dt
            self.total_number_elt_summed += 1
            self.total_acc_x += accel_rot[0]
            self.total_acc_y += accel_rot[1]
            self.total_acc_z += accel_rot[2] + ACCEL_G
            if (monotonic() - self.timestamp_last_summed_acc) * PUBLISH_SUMMED_MEASURE_ERROR_ACC > 1:
                self.logger.info("Sum dt : {}, Number of elements : {}, Sum Acc : {} "
                                 .format(self.total_time, self.total_number_elt_summed,
                                         [self.total_acc_x, self.total_acc_y, self.total_acc_z]))
                self.timestamp_last_summed_acc = monotonic()
        # Reduce the velocity to reduce the drift
        if METHOD_RESET_VELOCITY and (monotonic() - self.timestamp_last_reset_vel) * RESET_VEL_FREQ > 1:
            self.speed_x *= RESET_VEL_FREQ_COEF_X
            self.speed_y *= RESET_VEL_FREQ_COEF_Y
            self.speed_z *= RESET_VEL_FREQ_COEF_Z
            self.timestamp_last_reset_vel = monotonic()
        # Integrate to get the position
        self.pos.x += self.speed_x * dt
        self.pos.y += self.speed_y * dt
        self.pos.z += self.speed_z * dt
        if (monotonic() - self.timestamp_last_displayed_acc) * POSITION_PUBLISH_ACC_FREQ > 1:
            self.logger.info("Acceleration after rotation :  {}, Corrected speed : {} "
                             .format(accel_rot, [self.speed_x, self.speed_y, self.speed_z]))
            self.timestamp_last_displayed_acc = monotonic()

    def publish_to_visualization(self):
        if self.last_visualized_pos is None \
                or (monotonic() - self.last_visualized_pos.ts) * POSITION_PUBLISH_FREQ > 1:

            if self.pos != self.last_visualized_pos:
                self.publish("position", self.pos.__dict__, POS_VALIDITY_MS)
                self.last_visualized_pos = copy.deepcopy(self.pos)

    # DEBUG FUNCTIONS
    def debug_input_data(self, input_data, timestamp):
        if DEBUG_POSITION > 1:
            # Counter of valid values
            self.count_valid_input += 1

            # Difference in time from the timestamp of the data
            if self.timestamp_last_input == 0:
                timeDelta = 0
            else:
                timeDelta = timestamp - self.timestamp_last_input

            # Log output for each element received
            self.logger.info("Received Acceleration element N° {} within {} loops, time between samples : {}. "
                             .format(self.count_valid_input, self.countall, timeDelta))
            self.timestamp_last_input = timestamp

            if DEBUG_POSITION >= 3:
                self.logger.info("Data :  {}".format(input_data))

    def debug_downsample_publish(self, data_dict):
        if DEBUG_POSITION > 1:
            # Counter of valid values
            self.count_outputs += 1

            # Log output for each element sent
            self.logger.info("Sent data N° {}, time between samples : {:.4f} seconds. "
                             .format(self.count_outputs, monotonic() - self.timestamp_last_output))

            if DEBUG_POSITION >= 3:
                self.logger.info("Data sent :  {}".format(data_dict))
        # self.logger.info("Data sent :  {}".format(data_dict))

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
